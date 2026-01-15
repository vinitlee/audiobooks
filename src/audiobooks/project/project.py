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
from audiobooks.core.audio import AudioChapter, AudioProcessor
from audiobooks.utils import (
    not_none_dict,
    clean_dict,
    ensure_list,
    filter_dict,
    confirm,
    write_json_atomic,
    write_sound_atomic,
)

from .spec import (
    ProjectConfig,
    ProjectPaths,
    ProjectState,
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

    chapters: Optional[List[AudioChapter]] = None

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
        self.book.chapters = self.book.chapters[:5]  # DEBUG

        self.lexicon = Lexicon(self.config.lex_g2g_paths, self.config.lex_g2p_paths)

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

        for chapter in self.book.chapters:
            if not self.state.chapter_is_complete(chapter.index):
                self.generate_chapter_tts(chapter)
            else:
                logging.info(
                    f"Chapter @{chapter.index}, '{chapter.title}' already generated, skipping."
                )

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

    def generate_chapter_tts(self, chapter: Chapter):
        output_path = self.paths.chapter(chapter.index)

        chapter_bl = BlockList(chapter)

        audio_parts = self.processor.generate(chapter_bl, progress_title=chapter.title)
        full_audio = np.concat(audio_parts)

        write_sound_atomic(output_path, full_audio, self.processor.sample_rate)
        self.state.add_artifact(f"ch_{chapter.index}", output_path)
        if self.validate_chapter_tts(chapter):
            self.state.chapter_set_complete(chapter_number=chapter.index)
            self.commit_state()

    def validate_chapter_tts(self, chapter: Chapter):
        """
        Checks to verify that tts has completed successfully.
        """
        chapter_path = self.paths.chapter(chapter.index)
        return chapter_path.exists() and chapter_path.stat().st_size > 0

    # Process Audio

    def step_process_audio(self):

        if self.validate_step_process_audio():
            self.state.step_set_complete(StepName.PROCESS_AUDIO)
            self.commit_state()

    def validate_step_process_audio(self):
        """
        Checks to verify that process_audio has completed successfully.
        """
        return False

    # Build M4B

    def step_build_m4b(self):

        if self.validate_step_build_m4b():
            self.state.step_set_complete(StepName.BUILD_M4B)
            self.commit_state()

    def validate_step_build_m4b(self):
        """
        Checks to verify that build_m4b has completed successfully.
        """
        return False

    # Copy to library

    def step_copy_to_library(self):

        if self.validate_step_copy_to_library():
            self.state.step_set_complete(StepName.COPY_TO_LIBRARY)
            self.commit_state()

    def validate_step_copy_to_library(self):
        """
        Checks to verify that copy_to_library has completed successfully.
        """
        if self.config.output_path is None:
            return False

        return False

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

    # def step_index(self, step_name: StepName):
    #     return self.STEP_ORDER.index(step_name)

    # def make_splits(self):
    #     """
    #     For each chapter
    #         Generates TTS
    #         Saves split wav
    #         Log split data
    #     """
    #     print("\n### Making Splits ###")
    #     tts_processor = TTSProcessor(self.lexicon)
    #     self.complete["splits"] = True
    #     self.save_state()

    # def make_master(self):
    #     """
    #     Combine all raw splits into a compressed master file
    #     Process like normalizing volume, compression, etc
    #     """
    #     print("\n### Making Master ###")
    #     pass

    # def make_m4b(self):
    #     """
    #     Turns master file into m4b
    #     Generates needed helper files
    #     Saves m4b
    #     """
    #     print("\n### Making M4B ###")
    #     pass

    # def copy_to_library(self):
    #     """
    #     (Optional)
    #     Copies m4b (and epub) to the library location
    #     """
    #     pass

    # def run(self):
    #     self.make_splits()
    #     self.make_master()
    #     self.make_m4b()
    #     self.copy_to_library()

    # # ------------------------------------------------------------------------

    # # -- State --

    # def restore_from_dict(self, d: dict):
    #     pass

    # def to_dict(self):
    #     d = {
    #         # "title": None,
    #         # "author": None,
    #         # "series": None,
    #         # "progress": self.progress.to_dict(),
    #         # "book": {"epub": None},
    #         # "tts": {
    #         #     "voice": "am_michael",
    #         #     "speed": 1.0,
    #         # },
    #         # "lexicon": {
    #         #     "g2g": [],
    #         #     "g2p": [],
    #         # },
    #         # "output": None,
    #         # "saved": None,
    #     }

    #     return d

    # def get_state(self, path: str):
    #     return dpath.util.get(self.state, path, default=None)

    # def set_state(self, path: Union[str, list[str]], val: Any):
    #     return dpath.util.new(self.state, path, val)

    # def save_state(self):
    #     self.set_state("saved", datetime.datetime.now())
    #     yaml_stream = (self.dir / self.state_file).open("w", encoding="utf-8")
    #     yaml.dump(self.state, yaml_stream, allow_unicode=True, sort_keys=False)
    #     yaml_stream.close()

    #     # print(f"Saved state:\n{yaml.dump(self.state)}")

    # def load_state(self):
    #     linked = ["progress", "tts"]
    #     yaml_stream = (self.dir / self.state_file).open("r", encoding="utf-8")

    #     loaded = yaml.load(yaml_stream, yaml.FullLoader)
    #     self.state |= filter_dict(loaded, blacklist=linked)
    #     # Assign linked entries
    #     self.complete |= loaded.get("progress", {})
    #     self.set_state_tts(**filter_dict(loaded["tts"], whitelist=["voice", "speed"]))

    # # -- Book --

    # def init_book(self):
    #     epub_path = self.dir / self.epub
    #     self.book = Book(epub_path)
    #     state_overrides = {
    #         k: v
    #         for k in ["title", "author", "series"]
    #         if (v := self.state.get(k, None))
    #     }
    #     self.book.meta.add_overrides(state_overrides)

    #     (self.dir / self.fulltext).write_text(self.book.fulltext, encoding="utf-8")
    #     self.set_state_book_metadata(
    #         {
    #             "title": self.book.meta.title,
    #             "author": self.book.meta.author,
    #             "series": self.book.meta.series,
    #         }
    #     )

    # @property
    # def epub(self) -> str:
    #     fromstate = self.get_state("book/epub")
    #     if fromstate is None:
    #         found = self.find_epub()
    #         if found:
    #             self.epub = found
    #             return found
    #         else:
    #             raise Exception("EPUB could not be found")
    #     return str(fromstate)

    # @epub.setter
    # def epub(self, val):
    #     self.set_state("book/epub", val)

    # def find_epub(self):
    #     epubs = list(self.dir.glob("*.epub"))
    #     # If there is an EPUB in the directory
    #     if len(epubs):
    #         if len(epubs) > 1:
    #             logging.warning("There seem to be multiple EPUBs.")
    #         return epubs[0].name

    # def set_state_book_metadata(self, mapping: Optional[dict[str, str | None]] = None):
    #     if mapping is not None:
    #         valid_keys = ["title", "author", "series"]
    #         new_vals = filter_dict(mapping, whitelist=valid_keys)
    #         self.state.update(new_vals)

    # def get_state_book_metdata(self):
    #     valid_keys = ["title", "author", "series"]
    #     return {k: self.state[k] for k in valid_keys if self.state[k]}

    # # -- TTS --

    # @property
    # def tts_params(self):
    #     return self.state.get("tts", {})

    # def set_state_tts(self, voice: Optional[str] = None, speed: Optional[float] = None):
    #     newstate = filter_dict({"voice": voice, "speed": speed})
    #     self.state["tts"] = self.tts_params | newstate

    # # -- Lexicon --

    # def init_lexicon(
    #     self,
    #     g2g_paths: Optional[List[str]] = None,
    #     g2p_paths: Optional[List[str]] = None,
    # ):
    #     g2g_full_paths = []
    #     g2p_full_paths = []

    #     g2g_full_paths.extend(g2g_paths or [])
    #     g2p_full_paths.extend(g2p_paths or [])

    #     g2g_full_paths.extend(self.lexicon_sources.get("g2g", []))
    #     g2p_full_paths.extend(self.lexicon_sources.get("g2p", []))

    #     # Instantiate Lexicon
    #     self.lexicon = Lexicon(g2g_full_paths, g2p_full_paths)
    #     # TODO: Consider moving this to save_state
    #     self.set_state_lexicon_sources(
    #         self.lexicon.g2g_paths, self.lexicon.g2p_paths, append=False
    #     )

    # @property
    # def lexicon_sources(self):
    #     return self.state.get("lexicon", {})

    # def set_state_lexicon_sources(
    #     self, g2g: list[str] = [], g2p: list[str] = [], append=True
    # ):
    #     if append:
    #         # Keep old sources and just add the new ones
    #         current = self.lexicon_sources
    #         g2g += current.get("g2g", [])
    #         g2p += current.get("g2p", [])

    #     # Fully resolve all paths and store as strings
    #     g2g = [str(Path(p).expanduser().resolve()) for p in g2g]
    #     g2p = [str(Path(p).expanduser().resolve()) for p in g2p]

    #     # Only keep unique
    #     g2g = list(set(g2g))
    #     g2p = list(set(g2p))
    #     newstate = {
    #         "g2g": g2g,
    #         "g2p": g2p,
    #     }
    #     self.state["lexicon"] |= filter_dict(newstate)


# %%
