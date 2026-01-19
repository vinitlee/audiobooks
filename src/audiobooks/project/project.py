# %%
# Typing
from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Tuple,
    List,
    Dict,
    Union,
    Sequence,
    Self,
    Mapping,
    Literal,
    ClassVar,
)

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken

# Debugging
from line_profiler import profile

# Filesystem
from pathlib import Path
import shutil
from pathvalidate import sanitize_filepath
import glob

# OS
# import subprocess
# import shutil
# import argparse

# Text
# import re
import yaml
import json
import datetime

# Objects
import dpath.util

from dataclasses import dataclass, field, asdict
from enum import Enum

# from functools import lru_cache

# Math
import numpy as np
import cv2

# import cv2
# import torch

# Media
import soundfile as sf

# ffmpeg
# from ebooklib import epub
# import ebooklib
# import lxml.html

# UI
import warnings
from tqdm import tqdm
import traceback
import logging

# Project Libraries
from audiobooks.core.book import Book, Chapter
from audiobooks.tts.tts import Lexicon, TTSProcessor, KPipelineLazy, Voice, BlockList
from audiobooks.core.audio import (
    master_audio_from_chapters,
    m4b_from_master_audio,
    generate_ffmetadata,
)
from audiobooks.utils import (
    not_none_dict,
    clean_dict,
    ensure_list,
    filter_dict,
    confirm,
    write_json_atomic,
    write_sound_atomic,
    copy_atomic,
)

from .spec import (
    ProjectConfig,
    ProjectPaths,
    ProjectState,
    ChapterRecord,
    ChapterAudio,
    StepName,
    ProjectFlags,
    Result,
)


class AudiobookProject:
    config: ProjectConfig
    paths: ProjectPaths
    state: ProjectState

    book: Book

    # voice: Voice
    # speed: float
    lexicon: Lexicon

    processor: TTSProcessor

    # chapters: Optional[List[AudioChapter]] = None

    STEP_ORDER: ClassVar[list[StepName]] = [
        StepName.INIT,
        StepName.TTS,
        StepName.PROCESS_AUDIO,
        StepName.BUILD_M4B,
        StepName.COPY_TO_LIBRARY,
    ]

    STEP_FUNCTIONS: ClassVar[Mapping[StepName, str]] = {
        StepName.INIT: "step_init",
        StepName.TTS: "step_tts",
        StepName.PROCESS_AUDIO: "step_process_audio",
        StepName.BUILD_M4B: "step_build_m4b",
        StepName.COPY_TO_LIBRARY: "step_copy_to_library",
    }

    def __init__(
        self, config: ProjectConfig, paths: ProjectPaths, state: ProjectState
    ) -> None:
        self.config = config
        self.paths = paths
        self.state = state

    @classmethod
    def open(
        cls,
        init_path: Path | str,
        config: ProjectConfig,
    ):

        project_config: ProjectConfig
        project_paths: ProjectPaths
        restore_state: ProjectState

        open_result = None
        init_path = Path(init_path)
        logging.info(f"Opening from {init_path}")

        project_paths = ProjectPaths(init_path)
        # If the init_path looks like a preexisting project dir
        if project_paths.looks_like_project():
            # Load config from project.json
            project_config = ProjectConfig.from_source(project_paths.config)
            # Load state from state.json
            restore_state = ProjectState.from_source(project_paths.state)
            open_result = Result.RESUMED
        # If the init_path looks like an epub
        elif init_path.is_file() and init_path.suffix == ".epub":
            # Use config from arguments
            project_config = config
            # Make a new project and move epub into it
            project_paths = cls.new_project_dir(init_path)
            # Start with a clear state
            restore_state = ProjectState()
            # Make file records
            write_json_atomic(project_config.to_dict(), project_paths.config)
            write_json_atomic(restore_state.to_dict(), project_paths.state)
            open_result = Result.NEW_PROJECT
        else:
            open_result = Result.FAILED_INVALID_PATH
            logging.error(f"{init_path} is not a valid starting path.")
            raise

        obj = cls(config=project_config, paths=project_paths, state=restore_state)

        return obj

    @staticmethod
    def new_project_dir(source_path: Path) -> ProjectPaths:
        new_paths: ProjectPaths

        if not (source_path.suffix == ".epub" and source_path.is_file()):
            logging.error(f"{source_path} is invalid.")
            raise

        # Make dir
        newdir = source_path.parent / source_path.stem
        newdir = newdir.expanduser().resolve()
        if newdir.exists():
            logging.warning(f"{newdir} already exists.")
            replace = confirm(f"{newdir} exists. Delete and replace?")
            if not replace:
                raise
            else:
                shutil.rmtree(str(newdir))
        newdir.mkdir(parents=True, exist_ok=False)
        new_paths = ProjectPaths(newdir)

        # Move epub into it
        new_paths.book.parent.mkdir(parents=True)
        source_path.rename(new_paths.book)

        return new_paths

    def commit_state(self):
        """
        Atomic write state.json
        """
        write_json_atomic(self.state.to_dict(), self.paths.state)

    """
    Each stage is a function with:

    Inputs: project.config, project.state, and filesystem artifacts

    Outputs: artifacts + record files

    State transition: mark completion only after outputs are known-good
    """

    # Runtime Prep

    def prepare_runtime(self):
        logging.info("Preparing runtime environment")
        self.book = Book(self.paths.book)
        self.book.meta.add_overrides(**self.config.metadata_overrides())
        self.book.generate_chapters()
        self.book.chapters = self.book.chapters

        # TODO: Consider moving lexicon creation into TTSProcessor and just instantiating that here instead
        self.lexicon = Lexicon(self.config.lex_g2g_path, self.config.lex_g2p_path)

    # Init

    def step_init(self):
        """
        Set up project tree and set up objects that require fs access
        """
        self.paths.make_tree(exist_ok=True)
        self.paths.text.write_text(
            self.book.fulltext(),
            encoding="utf-8",
        )

        if self.validate_step_init():
            self.state.step_set_complete(StepName.INIT)
            self.commit_state()

    def validate_step_init(self):
        """
        Checks to verify that init has completed successfully.
        """
        text_real = self.paths.text.exists() and self.paths.text.stat().st_size > 0
        dirs_exist = all([d.exists() for d in self.paths.dirs])
        return all([text_real, dirs_exist])

    # TTS

    def step_tts(self):
        """
        Generate TTS WAVs for each chapter in the book.
        Main bulk of the work.
        """

        self.processor = TTSProcessor(
            self.lexicon,
            self.config.tts_voice,
            self.config.tts_speed,
        )

        chapter_record = ChapterRecord()
        if self.paths.chapter_record.exists():
            chapter_record = ChapterRecord.from_source(self.paths.chapter_record)

        for chapter in self.book.chapters:
            if not self.state.chapter_is_complete(chapter.index):
                ch = self.generate_chapter_tts(chapter)
                chapter_record.add_chapter(ch)
                chapter_record.dump(self.paths.chapter_record)
            else:
                logging.info(
                    f"Chapter @{chapter.index}, '{chapter.title}' already generated, skipping."
                )

        # Save unknown words
        self.lexicon.amend_g2p(self.processor.unknown_words)

        if self.validate_step_tts():
            self.state.step_set_complete(StepName.TTS)
            self.commit_state()

    def validate_step_tts(self):
        """
        Checks to verify that tts has completed successfully.
        """
        all_chapters_complete = all(
            [self.state.chapter_is_complete(c.index) for c in self.book.chapters]
        )
        all_chapters_exist = all(
            [self.paths.chapter(c.index).exists() for c in self.book.chapters]
        )
        return all([all_chapters_complete, all_chapters_exist])

    def generate_chapter_tts(self, chapter: Chapter) -> ChapterAudio:
        output_path = self.paths.chapter(chapter.index)

        chapter_bl = BlockList(chapter)

        audio_parts = self.processor.generate(chapter_bl, progress_title=chapter.title)
        full_audio = np.concat(audio_parts)

        write_sound_atomic(output_path, full_audio, self.processor.sample_rate)
        self.state.add_artifact(f"ch_{chapter.index}", output_path)
        if self.validate_chapter_tts(chapter):
            self.state.chapter_set_complete(chapter_number=chapter.index)
            self.commit_state()

            return ChapterAudio(
                chapter.title,
                chapter.index,
                len(full_audio) / self.processor.sample_rate,
            )
        raise AssertionError(f"generate_chapter_tts for {chapter} failed to validate")

    def validate_chapter_tts(self, chapter: Chapter):
        """
        Checks to verify that tts has completed successfully.
        """
        chapter_path = self.paths.chapter(chapter.index)
        return chapter_path.exists() and chapter_path.stat().st_size > 0

    # Process Audio

    def step_process_audio(self):

        # Combine all the previously generated chapter audio files into a master m4a

        chapter_record = ChapterRecord.from_source(self.paths.chapter_record)
        audio_paths = [self.paths.chapter(ch.index) for ch in chapter_record]
        output_path = self.paths.master_audio
        master_audio_from_chapters(audio_paths, output_path)
        self.state.add_artifact("master_audio", output_path)

        if self.validate_step_process_audio():
            self.state.step_set_complete(StepName.PROCESS_AUDIO)
            self.commit_state()

    def validate_step_process_audio(self):
        """
        Checks to verify that process_audio has completed successfully.
        """
        master_audio_exists = self.paths.master_audio.exists()
        master_audio_nonzero = self.paths.master_audio.stat().st_size > 0
        return master_audio_exists and master_audio_nonzero

    # Build M4B

    def step_build_m4b(self):

        chapter_record = ChapterRecord.from_source(self.paths.chapter_record)
        # Generate ffmetadata from ChapterRecord
        generate_ffmetadata(
            self.paths.ffmetadata,
            str(self.book.meta.title),
            str(self.book.meta.author),
            str(self.book.meta.year),
            chapter_record,
        )

        # Grab Cover Image
        cover_img = self.book.meta.cover
        cv2.imwrite(str(self.paths.cover_img), cover_img)

        # Convert m4a + ffmetadata + cover -> m4b
        m4b_from_master_audio(
            self.paths.master_audio,
            self.paths.cover_img,
            self.paths.ffmetadata,
            self.paths.audiobook,
        )

        if self.validate_step_build_m4b():
            self.state.step_set_complete(StepName.BUILD_M4B)
            self.commit_state()

    def validate_step_build_m4b(self):
        """
        Checks to verify that build_m4b has completed successfully.
        """
        m4b_exists = self.paths.audiobook.exists()
        m4b_nonzero = self.paths.audiobook.stat().st_size > 0
        return m4b_exists and m4b_nonzero

    # Copy to library

    def step_copy_to_library(self):
        if self.config.output_path is None:
            return False

        title = str(self.book.meta.title)[:255].strip()

        output_path_base = self.config.output_path / title / title
        output_path_base.parent.mkdir(parents=True, exist_ok=True)

        output_path_m4b = output_path_base.with_suffix(self.paths.audiobook.suffix)
        output_path_epub = output_path_base.with_suffix(self.paths.book.suffix)

        copy_atomic(self.paths.audiobook, output_path_m4b)
        self.state.add_artifact("library_m4b", output_path_m4b)
        copy_atomic(self.paths.book, output_path_epub)
        self.state.add_artifact("library_epub", output_path_epub)

        if self.validate_step_copy_to_library():
            self.state.step_set_complete(StepName.COPY_TO_LIBRARY)
            self.commit_state()

    def validate_step_copy_to_library(self):
        """
        Checks to verify that copy_to_library has completed successfully.
        """
        if self.config.output_path is None:
            logging.info("No output path set.")
            return False

        output_path_m4b = self.state.artifacts.get("library_m4b")
        output_path_epub = self.state.artifacts.get("library_epub")

        if output_path_m4b is None or output_path_epub is None:
            return False

        return output_path_m4b.exists() and output_path_epub.exists()

    def run_all(self):
        self.prepare_runtime()

        for step in self.STEP_ORDER:
            if self.config.flag_init_only and step != StepName.INIT:
                logging.info(f"Init only, skipping.")
                break
            if self.state.step_is_complete(step):
                logging.info(f"Skipping step {step.name}")
                continue

            logging.info(f"Running step {step.name}")
            step_fn = getattr(self, self.STEP_FUNCTIONS[step])
            step_fn()

            if not self.state.step_is_complete(step):
                logging.error(f"{step.name} was not successful.")
                break

        logging.info("Done!")
