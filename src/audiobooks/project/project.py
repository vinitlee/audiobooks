# %%
# Typing
from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken
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
    )

# Debugging
from line_profiler import profile

# Filesystem
from pathlib import Path
from pathvalidate import sanitize_filepath
import glob

# OS
# import subprocess
# import shutil
# import argparse

# Text
# import re
import yaml
import datetime

# Objects
import dpath.util

from dataclasses import dataclass, field
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
from audiobooks.utils.utils import not_none_dict, clean_dict, ensure_list, filter_dict


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
class ProjectConfig:
    init_path: str
    tts_voice: Optional[Voice] = None
    tts_speed: Optional[float] = None
    lex_g2g_paths: Optional[List[str]] = None
    lex_g2p_paths: Optional[List[str]] = None
    override_author: Optional[str] = None
    override_series: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class TtsConfig:
    voice: Optional[Voice] = None
    speed: Optional[float] = None


@dataclass
class LexiconConfig:
    # XXX: Consider making everything a Path
    g2g: list[str] = field(default_factory=list)
    g2p: list[str] = field(default_factory=list)


@dataclass
class ProjectSpec:
    project_dir: Path
    epub_path: Optional[Path] = None

    tts_voice: Optional[Voice] = None
    tts_speed: Optional[float] = None
    lex_g2g_paths: Optional[list[str]] = None
    lex_g2p_paths: Optional[list[str]] = None
    override_author: Optional[str] = None
    override_series: Optional[str] = None
    output_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: Union[Path, str]):
        yaml_path = Path(yaml_path)
        d = yaml.safe_load(yaml_path.open(encoding="utf-8"))

        return cls(
            project_dir=yaml_path.parent,
            epub_path=d.get("epub"),
            tts_voice=d.get("voice"),
            tts_speed=d.get("speed"),
            lex_g2g_paths=d.get("g2g"),
            lex_g2p_paths=d.get("g2p"),
            override_author=d.get("author"),
            override_series=d.get("series"),
            output_path=d.get("output_path"),
        )

    @classmethod
    def resolve(
        cls,
        patch: Optional[ProjectSpec],
        snapshot: Optional[ProjectSpec],
        result: Result,
    ) -> ProjectSpec:
        if patch is None and snapshot is None:
            return cls()
        return cls()

    @property
    def params(self):
        return {}


@dataclass
class ProjectState:
    splits: list[AudioChapter]
    files: set[str]


class AudiobookProject:
    project_dir: Path
    _open_result: str

    book: Book
    voice: Voice
    lexicon: Lexicon

    processor: TTSProcessor

    chapters: Optional[List[AudioChapter]] = None

    # Paths
    state_file: str = "project.yaml"
    fulltext: str = "fulltext.txt"
    cover: str = "cover.jpg"
    wavfiles: str = "files"
    ffmeta: str = "ffmetadata"

    def __init__(self, spec: ProjectSpec) -> None:
        self.project_dir = spec.project_dir
        self.book = Book(epub_path)
        self.voice = tts_voice if tts_voice is not None else "am_michael"
        self.speed = tts_speed if tts_speed is not None else 1.0
        self.lexicon = Lexicon(lex_g2g_paths, lex_g2p_paths)
        self.book.meta.override_author = override_author
        self.book.meta.override_series = override_series
        self.output_path = output_path

    @classmethod
    def open(
        cls,
        init_path: Path | str,
        config: ProjectConfig,
    ):

        logging.info(f"Opening from {init_path}")

        open_result = None

        snapshot: Optional[ProjectSpec] = None
        restore_state: ProjectState

        init_path = Path(init_path)
        if init_path.is_dir() and (epubs := list(init_path.glob("*.epub"))):
            # Is a potential project dir with an EPUB inside
            if (state_file := (init_path / cls.state_file)).exists():
                # project.yaml file exists
                try:
                    snapshot = ProjectSpec.from_yaml(state_file)
                    # runtime_state = AudiobookProjectState()
                    open_result = Result.RESUMED
                except Exception as e:
                    logging.error(e)
                    # Could not load from project.yaml, ignore state and start fresh
                    if len(epubs) == 1:
                        snapshot = ProjectSpec(
                            project_dir=init_path,
                            epub_path=epubs[0],
                        )
                        open_result = Result.RESET_FROM_CORRUPTION
                    else:
                        open_result = Result.FAILED_MULTIPLE_EPUBS
            else:
                snapshot = ProjectSpec(
                    project_dir=init_path,
                    epub_path=epubs[0],
                )
                open_result = Result.NEW_FROM_EXISTING_DIR
        elif init_path.is_file() and init_path.suffix == ".epub":
            # Is an EPUB
            try:
                project_dir, epub_path = cls.new_project_dir(init_path)
                snapshot = ProjectSpec(project_dir=project_dir, epub_path=epub_path)
                open_result = Result.NEW_PROJECT
            except Exception as e:
                logging.error(e)

        if open_result is None:
            # Does not look compatible
            open_result = Result.FAILED_INVALID_PATH
            logging.exception(f"{init_path} is not a valid starting path.")
            raise

        # effective = AudiobookProjectSpec.resolve(
        #     patch=patch, snapshot=snapshot, result=open_result
        # )

        obj = cls(**effective.params)
        # if open_result == Result.RESUMED:
        #     obj.rehydrate(runtime_state)

        return obj

    def rehydrate(self, state: ProjectState):
        pass

    @staticmethod
    def new_project_dir(epub_path: Path) -> tuple[Path, Path]:
        if not (epub_path.suffix == ".epub" and epub_path.exists()):
            logging.exception(f"{epub_path} is invalid.")
            raise

        # Make dir
        newdir = epub_path.parent / epub_path.stem
        newdir = newdir.expanduser().resolve()
        if newdir.exists():
            logging.exception(f"{newdir} already exists.")
            raise
        newdir.mkdir(parents=True, exist_ok=False)

        # Move epub into it
        new_epub_path = newdir / epub_path.name
        epub_path.rename(new_epub_path)

        return newdir, new_epub_path

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
