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
# import numpy as np
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
from audiobooks.core.book import Book
from audiobooks.tts.tts import Lexicon, TTSProcessor, KPipelineLazy, Voice
from audiobooks.core.audio import AudioChapter, AudioProcessor
from audiobooks.utils import (
    not_none_dict,
    clean_dict,
    ensure_list,
    filter_dict,
    confirm,
)


# %%
class Result(Enum):
    NEW_PROJECT = 1
    NEW_FROM_EXISTING_DIR = 5
    RESUMED = 2
    RESET_FROM_CORRUPTION = 3
    FAILED_INVALID_PATH = 4
    FAILED_MULTIPLE_EPUBS = 6


# %%


@dataclass
class TtsConfig:
    voice: Optional[Voice] = None
    speed: Optional[float] = None


@dataclass
class LexiconConfig:
    # XXX: Consider making everything a Path
    g2g: list[str] = field(default_factory=list)
    g2p: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path

    config_name: str = "project.json"
    state_name: str = "state.json"
    master_audio_name: str = "master.m4a"
    audiobook_name: str = "final.m4b"
    chapter_record_name: str = "chapters.json"
    text_name: str = "book.txt"
    book_name: str = "book.epub"
    cover_name: str = "cover.jpg"

    source_dir_name: str = "source"
    records_dir_name: str = "records"
    artifacts_dir_name: str = "artifacts"
    extras_dir_name: str = "extras"

    chapters_dir_name: str = "chapter_wavs"
    chapter_template: str = "chapter_{}.wav"

    """
    project_dir/
        my_epub.epub
        project.json
        state.json
        source/
            book.epub
        records/
            ?
        artifacts/
            extras/
                cover.jpg
            chapter_wavs/
                chapter_0.wav
                ...
            master.m4a
            final.m4b
    """

    @property
    def config(self) -> Path:
        return self.project_dir / self.config_name

    @property
    def state(self) -> Path:
        return self.project_dir / self.state_name

    @property
    def source_dir(self) -> Path:
        return self.project_dir / self.source_dir_name

    @property
    def book(self) -> Path:
        return self.source_dir / self.book_name

    @property
    def records_dir(self) -> Path:
        return self.project_dir / self.records_dir_name

    @property
    def chapter_record(self) -> Path:
        return self.records_dir / self.chapter_record_name

    @property
    def artifacts_dir(self) -> Path:
        return self.project_dir / self.artifacts_dir_name

    @property
    def extras_dir(self) -> Path:
        return self.artifacts_dir / self.extras_dir_name

    @property
    def cover_img(self) -> Path:
        return self.extras_dir / self.cover_name

    @property
    def chapters_dir(self) -> Path:
        return self.artifacts_dir / self.chapters_dir_name

    def chapter(self, chapter_number: int) -> Path:
        return self.chapters_dir / self.chapter_template.format(chapter_number)

    @property
    def master_audio(self) -> Path:
        return self.artifacts_dir / self.master_audio_name

    @property
    def audiobook(self) -> Path:
        return self.artifacts_dir / self.audiobook_name

    @property
    def text(self) -> Path:
        return self.artifacts_dir / self.text_name

    def looks_like_project(self) -> bool:
        return self.project_dir.is_dir() and self.book.exists() and self.config.exists()

    def make_tree(self):
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.extras_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    tts_voice: Voice = "am_michael"
    tts_speed: float = 1.0
    lex_g2g_paths: Optional[list[str]] = None
    lex_g2p_paths: Optional[list[str]] = None
    override_author: Optional[str] = None
    override_series: Optional[str] = None
    output_path: Optional[str] = None

    def metadata_overrides(self, none_ok=False) -> Mapping[str, str | None]:
        d = {
            "author": self.override_author,
            "series": self.override_series,
        }
        if not none_ok:
            d = not_none_dict(d)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProjectConfig:
        valid_keys = [
            "tts_voice",
            "tts_speed",
            "lex_g2g_paths",
            "lex_g2p_paths",
            "override_author",
            "override_series",
            "output_path",
        ]
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_source(cls, path: Path | str) -> ProjectConfig:
        path = Path(path)
        d = json.load(path.open("r", encoding="utf-8"))
        return cls.from_dict(d)

    def dump(self, path: Path | str):
        json.dump(
            asdict(self),
            Path(path).open("w", encoding="utf-8"),
            indent=4,
        )


# StepName = Literal["INIT", "TTS", "PROCESS_AUDIO", "BUILD_M4B", "COPY_TO_LIBRARY"]
class StepName(Enum):
    INIT = "initialization"
    TTS = "tts generation"
    PROCESS_AUDIO = "audio processing"
    BUILD_M4B = "building m4b"
    COPY_TO_LIBRARY = "copying"


ChapterStatus = Literal["NOT_STARTED", "IN_PROGRESS", "FINISHED"]


@dataclass
class ProjectState:
    completed_steps: set[StepName] = field(default_factory=set)
    completed_chapters: Mapping[int, ChapterStatus] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    updated_at: str = ""

    def __post_init__(self):
        self.update_time()

    def update_time(self):
        self.updated_at = datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p")

    @classmethod
    def from_source(cls, path: Path | str):
        d = json.load(Path(path).open("r", encoding="utf-8"))
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_steps": [s.value for s in self.completed_steps],
            "completed_chapters": self.completed_chapters,
            "artifacts": self.artifacts,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d) -> ProjectState:
        return cls(
            completed_steps={StepName(s) for s in d["completed_steps"]},
            completed_chapters=d["completed_chapters"],
            artifacts=d["artifacts"],
            updated_at=d["updated_at"],
        )

    def dump(self, path: Path | str):
        json.dump(
            self.to_dict(),
            Path(path).open("w", encoding="utf-8"),
            indent=4,
        )


class AudiobookProject:
    paths: ProjectPaths
    state: ProjectState

    book: Book

    voice: Voice
    speed: float
    lexicon: Lexicon

    output_path: Path | None = None

    processor: TTSProcessor

    chapters: Optional[List[AudioChapter]] = None

    def __init__(
        self, config: ProjectConfig, paths: ProjectPaths, state: ProjectState
    ) -> None:
        self.paths = paths
        self.state = state

        self.book = Book(self.paths.book)
        self.book.meta.add_overrides(**config.metadata_overrides())

        self.voice = config.tts_voice
        self.speed = config.tts_speed
        self.lexicon = Lexicon(config.lex_g2g_paths, config.lex_g2p_paths)

        if config.output_path is not None:
            self.output_path = Path(config.output_path)

        self.state.completed_steps.add(StepName.INIT)
        self.state.dump(self.paths.state)

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
            project_config.dump(project_paths.config)
            restore_state.dump(project_paths.state)
            project_paths.make_tree()
            open_result = Result.NEW_PROJECT
        else:
            open_result = Result.FAILED_INVALID_PATH
            logging.exception(f"{init_path} is not a valid starting path.")
            raise

        obj = cls(config=project_config, paths=project_paths, state=restore_state)

        return obj

    """
    Each stage is a function with:

    Inputs: project.config, project.state, and filesystem artifacts

    Outputs: artifacts + record files

    State transition: mark completion only after outputs are known-good
    """

    def rehydrate(self, state: ProjectState):
        pass

    @staticmethod
    def new_project_dir(source_path: Path) -> ProjectPaths:
        new_paths: ProjectPaths

        if not (source_path.suffix == ".epub" and source_path.is_file()):
            logging.exception(f"{source_path} is invalid.")
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

    def init_dir(self, init_path):
        logging.info("Initializing Project")
        init_path = Path(init_path)

        if init_path.is_dir():
            set_up = False
            self.dir = init_path
            if (self.dir / self.state_file).exists():
                try:
                    # If there is a project.yaml
                    self.load_state()
                    set_up = True
                    logging.debug("Loaded YAML")
                except Exception as e:
                    traceback.print_exc()
                    set_up = False
                    logging.debug("Failed to load YAML")
            if not set_up:
                # Fallback if loading project.yaml fails; it will be overwritten
                found = self.find_epub()
                if found:
                    self.epub = found
                    logging.debug("Loaded as a fresh project")
                else:
                    raise Exception(
                        f"{init_path} is a directory but does not contain an EPUB."
                    )
        elif init_path.is_file():
            if init_path.suffix == ".epub":
                # If EPUB FILE
                # Make project dir
                self.dir = init_path.parent / init_path.stem
                self.dir = self.dir.resolve()
                self.dir.mkdir(parents=True, exist_ok=False)

                # Move epub into it
                self.epub = init_path.name
                init_path.rename(self.dir / self.epub)
            else:
                raise Exception(f"{init_path} is not an EPUB")
        else:
            raise Exception(f"{init_path} is not a valid starting path.")

    def make_splits(self):
        """
        For each chapter
            Generates TTS
            Saves split wav
            Log split data
        """
        print("\n### Making Splits ###")
        tts_processor = TTSProcessor(self.lexicon)
        self.complete["splits"] = True
        self.save_state()

    def make_master(self):
        """
        Combine all raw splits into a compressed master file
        Process like normalizing volume, compression, etc
        """
        print("\n### Making Master ###")
        pass

    def make_m4b(self):
        """
        Turns master file into m4b
        Generates needed helper files
        Saves m4b
        """
        print("\n### Making M4B ###")
        pass

    def copy_to_library(self):
        """
        (Optional)
        Copies m4b (and epub) to the library location
        """
        pass

    def run(self):
        self.make_splits()
        self.make_master()
        self.make_m4b()
        self.copy_to_library()

    # ------------------------------------------------------------------------

    # -- State --

    def restore_from_dict(self, d: dict):
        pass

    def to_dict(self):
        d = {
            # "title": None,
            # "author": None,
            # "series": None,
            # "progress": self.progress.to_dict(),
            # "book": {"epub": None},
            # "tts": {
            #     "voice": "am_michael",
            #     "speed": 1.0,
            # },
            # "lexicon": {
            #     "g2g": [],
            #     "g2p": [],
            # },
            # "output": None,
            # "saved": None,
        }

        return d

    def get_state(self, path: str):
        return dpath.util.get(self.state, path, default=None)

    def set_state(self, path: Union[str, list[str]], val: Any):
        return dpath.util.new(self.state, path, val)

    def save_state(self):
        self.set_state("saved", datetime.datetime.now())
        yaml_stream = (self.dir / self.state_file).open("w", encoding="utf-8")
        yaml.dump(self.state, yaml_stream, allow_unicode=True, sort_keys=False)
        yaml_stream.close()

        # print(f"Saved state:\n{yaml.dump(self.state)}")

    def load_state(self):
        linked = ["progress", "tts"]
        yaml_stream = (self.dir / self.state_file).open("r", encoding="utf-8")

        loaded = yaml.load(yaml_stream, yaml.FullLoader)
        self.state |= filter_dict(loaded, blacklist=linked)
        # Assign linked entries
        self.complete |= loaded.get("progress", {})
        self.set_state_tts(**filter_dict(loaded["tts"], whitelist=["voice", "speed"]))

    # -- Book --

    def init_book(self):
        epub_path = self.dir / self.epub
        self.book = Book(epub_path)
        state_overrides = {
            k: v
            for k in ["title", "author", "series"]
            if (v := self.state.get(k, None))
        }
        self.book.meta.add_overrides(state_overrides)

        (self.dir / self.fulltext).write_text(self.book.fulltext, encoding="utf-8")
        self.set_state_book_metadata(
            {
                "title": self.book.meta.title,
                "author": self.book.meta.author,
                "series": self.book.meta.series,
            }
        )

    @property
    def epub(self) -> str:
        fromstate = self.get_state("book/epub")
        if fromstate is None:
            found = self.find_epub()
            if found:
                self.epub = found
                return found
            else:
                raise Exception("EPUB could not be found")
        return str(fromstate)

    @epub.setter
    def epub(self, val):
        self.set_state("book/epub", val)

    def find_epub(self):
        epubs = list(self.dir.glob("*.epub"))
        # If there is an EPUB in the directory
        if len(epubs):
            if len(epubs) > 1:
                logging.warning("There seem to be multiple EPUBs.")
            return epubs[0].name

    def set_state_book_metadata(self, mapping: Optional[dict[str, str | None]] = None):
        if mapping is not None:
            valid_keys = ["title", "author", "series"]
            new_vals = filter_dict(mapping, whitelist=valid_keys)
            self.state.update(new_vals)

    def get_state_book_metdata(self):
        valid_keys = ["title", "author", "series"]
        return {k: self.state[k] for k in valid_keys if self.state[k]}

    # -- TTS --

    @property
    def tts_params(self):
        return self.state.get("tts", {})

    def set_state_tts(self, voice: Optional[str] = None, speed: Optional[float] = None):
        newstate = filter_dict({"voice": voice, "speed": speed})
        self.state["tts"] = self.tts_params | newstate

    # -- Lexicon --

    def init_lexicon(
        self,
        g2g_paths: Optional[List[str]] = None,
        g2p_paths: Optional[List[str]] = None,
    ):
        g2g_full_paths = []
        g2p_full_paths = []

        g2g_full_paths.extend(g2g_paths or [])
        g2p_full_paths.extend(g2p_paths or [])

        g2g_full_paths.extend(self.lexicon_sources.get("g2g", []))
        g2p_full_paths.extend(self.lexicon_sources.get("g2p", []))

        # Instantiate Lexicon
        self.lexicon = Lexicon(g2g_full_paths, g2p_full_paths)
        # TODO: Consider moving this to save_state
        self.set_state_lexicon_sources(
            self.lexicon.g2g_paths, self.lexicon.g2p_paths, append=False
        )

    @property
    def lexicon_sources(self):
        return self.state.get("lexicon", {})

    def set_state_lexicon_sources(
        self, g2g: list[str] = [], g2p: list[str] = [], append=True
    ):
        if append:
            # Keep old sources and just add the new ones
            current = self.lexicon_sources
            g2g += current.get("g2g", [])
            g2p += current.get("g2p", [])

        # Fully resolve all paths and store as strings
        g2g = [str(Path(p).expanduser().resolve()) for p in g2g]
        g2p = [str(Path(p).expanduser().resolve()) for p in g2p]

        # Only keep unique
        g2g = list(set(g2g))
        g2p = list(set(g2p))
        newstate = {
            "g2g": g2g,
            "g2p": g2p,
        }
        self.state["lexicon"] |= filter_dict(newstate)


# %%
