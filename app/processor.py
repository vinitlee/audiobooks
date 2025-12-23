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
import subprocess
import shutil
import argparse

# Text
import re
import yaml
import datetime

# Objects
import dpath.util
from dataclasses import dataclass
from functools import lru_cache

# Math
import numpy as np
import cv2
import torch

# Media
import soundfile as sf

# ffmpeg
from ebooklib import epub
import ebooklib
import lxml.html

# UI
import warnings
from tqdm import tqdm
import traceback

# Project Libraries
from book import Book
from tts import Lexicon, TTSProcessor, KPipelineLazy
from audio import AudioChapter, AudioProcessor

# %%


class AudiobookProject:
    book: Book
    lexicon: Lexicon
    override_metadata: Dict[str, Any] = {}
    # Generation
    processor: TTSProcessor
    # Paths

    statestore: str = "project.yaml"
    fulltext: str = "fulltext.txt"
    cover: str = "cover.jpg"
    wavfiles: str = "files"
    ffmeta: str = "ffmetadata"

    complete: dict[str, bool] = {
        "init": False,
        "splits": False,
        "join": False,
        "m4b": False,
        "copy": False,
    }

    state: dict[str, Any] = {
        "title": None,
        "author": None,
        "series": None,
        "progress": complete,
        "book": {"epub": None},
        "tts": {
            "voice": "am_michael",
            "speed": 1.0,
        },
        "lexicon": {
            "g2g": [],
            "g2p": [],
        },
        "saved": None,
    }

    chapters: list[AudioChapter] = []

    def __init__(
        self,
        init_path: Path | str,
        tts_voice: str | None = None,
        tts_speed: float | None = None,
        lex_g2g_paths: list[str] = [],
        lex_g2p_paths: list[str] = [],
        override_metadata: Dict[str, Any] = {},
    ) -> None:

        # Load an old project or make dir for new project
        self.init_dir(init_path)

        # Make Book
        self.set_state_book_metadata(override_metadata)
        self.init_book()

        # Make Lexicon
        self.init_lexicon(lex_g2g_paths, lex_g2p_paths)

        # Store TTS parameters
        self.set_state_tts(tts_voice, tts_speed)

        # Mark init completed and save progress
        self.complete["init"] = True
        self.save_state()

    def init_dir(self, init_path):
        init_path = Path(init_path)

        if init_path.is_dir():
            # If DIR
            set_up = False
            self.dir = init_path
            if (self.dir / self.statestore).exists():
                try:
                    # If there is a project.yaml
                    self.load_state()
                    set_up = True
                    print("Loaded YAML")
                except Exception as e:
                    traceback.print_exc()
                    set_up = False
                    print("Failed to load YAML")
            if not set_up:
                # Fallback if loading project.yaml fails; it will be overwritten
                found = self.find_epub()
                if found:
                    self.epub = found
                    print("Loaded as a fresh project")
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
        tts_processor = TTSProcessor(self.lexicon)
        self.complete["splits"] = True
        self.save_state()

    def make_master(self):
        """
        Combine all raw splits into a compressed master file
        Process like normalizing volume, compression, etc
        """
        pass

    def make_m4b(self):
        """
        Turns master file into m4b
        Generates needed helper files
        Saves m4b
        """
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

    @staticmethod
    def valid_dict(
        d: dict,
        whitelist: Optional[list[Any]] = None,
        blacklist: Optional[list[Any]] = None,
    ):
        def is_valid(k, v):
            non_null = v is not None
            white = True
            if whitelist is not None:
                white = k in whitelist
            black = True
            if blacklist is not None:
                black = k not in blacklist
            return non_null and white and black

        return {k: v for k, v in d.items() if is_valid(k, v)}

    # -- State --

    def get_state(self, path: str):
        return dpath.util.get(self.state, path, default=None)

    def set_state(self, path: Union[str, list[str]], val: Any):
        return dpath.util.new(self.state, path, val)

    def save_state(self):
        self.set_state("saved", datetime.datetime.now())
        yaml_stream = (self.dir / self.statestore).open("w", encoding="utf-8")
        yaml.dump(self.state, yaml_stream, allow_unicode=True, sort_keys=False)
        yaml_stream.close()

    def load_state(self):
        linked = ["progress", "tts"]
        yaml_stream = (self.dir / self.statestore).open("r", encoding="utf-8")

        loaded = yaml.load(yaml_stream, yaml.FullLoader)
        self.state |= self.valid_dict(loaded, blacklist=linked)
        # Assign linked entries
        self.complete |= loaded.get("progress", {})
        self.set_state_tts(
            **self.valid_dict(loaded["tts"], whitelist=["voice", "speed"])
        )

    # -- Book --

    def init_book(self):
        epub_path = self.dir / self.epub
        self.book = Book(epub_path)
        state_overrides = {
            k: self.state[k] for k in ["title", "author", "series"] if self.state[k]
        }
        self.book.meta._overrides |= state_overrides

        (self.dir / self.fulltext).write_text(self.book.fulltext)
        self.set_state_book_metadata(
            {
                "title": self.book.meta.title,
                "author": self.book.meta.creator,
                "series": self.book.meta.series,
            }
        )

        self.save_state()

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
                warnings.warn("There seem to be multiple EPUBs.")
            return epubs[0].name

    def set_state_book_metadata(self, mapping: dict[str, str | None]):
        valid_keys = ["title", "author", "series"]
        self.state.update(self.valid_dict(mapping, whitelist=valid_keys))

    def get_state_book_metdata(self):
        valid_keys = ["title", "author", "series"]
        return {k: self.state[k] for k in valid_keys if self.state[k]}

    # -- TTS --

    @property
    def tts_params(self):
        return self.state.get("tts", {})

    def set_state_tts(self, voice: Optional[str] = None, speed: Optional[float] = None):
        newstate = self.valid_dict({"voice": voice, "speed": speed})
        self.state["tts"] = self.tts_params | newstate

    # -- Lexicon --

    def init_lexicon(
        self,
        g2g_paths: list[str] = [],
        g2p_paths: list[str] = [],
    ):
        g2g_paths.extend(self.lexicon_sources.get("g2g", []))
        g2p_paths.extend(self.lexicon_sources.get("g2p", []))
        # Instantiate Lexicon
        self.lexicon = Lexicon(g2g_paths, g2p_paths)
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
        self.state["lexicon"] |= self.valid_dict(newstate)
