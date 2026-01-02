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
    write_json_atomic,
)


class Result(Enum):
    NEW_PROJECT = 1
    NEW_FROM_EXISTING_DIR = 5
    RESUMED = 2
    RESET_FROM_CORRUPTION = 3
    FAILED_INVALID_PATH = 4
    FAILED_MULTIPLE_EPUBS = 6


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

    @property
    def dirs(self) -> list[Path]:
        return [
            self.source_dir,
            self.extras_dir,
            self.records_dir,
            self.artifacts_dir,
            self.chapters_dir,
        ]

    def make_tree(self, exist_ok=True):
        self.source_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.extras_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.records_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.artifacts_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.chapters_dir.mkdir(parents=True, exist_ok=exist_ok)


@dataclass
class ProjectConfig:
    tts_voice: Voice = "am_michael"
    tts_speed: float = 1.0
    lex_g2g_paths: Optional[list[str]] = None
    lex_g2p_paths: Optional[list[str]] = None
    override_author: Optional[str] = None
    override_series: Optional[str] = None
    output_path: Optional[str] = None
    flag_init_only: bool = False

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
    def from_source(cls, path: Path) -> ProjectConfig:
        d = json.load(path.open("r", encoding="utf-8"))
        return cls.from_dict(d)

    def to_dict(self):
        return asdict(self)


class StepName(Enum):
    INIT = 1
    TTS = 2
    PROCESS_AUDIO = 3
    BUILD_M4B = 4
    COPY_TO_LIBRARY = 5


@dataclass(frozen=True)
class ProjectFlags:
    init_only: bool = False


@dataclass
class ProjectState:
    completed_steps: set[StepName] = field(default_factory=set)
    completed_chapters: set[int] = field(default_factory=set)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    updated_at: str = ""

    def __post_init__(self):
        self.update_time()

    def update_time(self):
        self.updated_at = datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p")

    def step_is_complete(self, step_name: StepName):
        return step_name in self.completed_steps

    def chapter_is_complete(self, chapter_number: int):
        return chapter_number in self.completed_chapters

    def step_set_complete(self, step_name: StepName):
        self.completed_steps.add(step_name)
        self.update_time()

    def chapter_set_complete(self, chapter_number: int):
        self.completed_chapters.add(chapter_number)
        self.update_time()

    def add_artifact(self, key: str, path: Path):
        if not path.exists():
            logging.warning(f"{path} added to artifacts but does not exist.")
        self.artifacts[key] = path
        self.update_time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_steps": [s.name for s in self.completed_steps],
            "completed_chapters": list(self.completed_chapters),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d) -> ProjectState:
        return cls(
            completed_steps={StepName[sn] for sn in d["completed_steps"]},
            completed_chapters=set(d["completed_chapters"]),
            artifacts={k: Path(v) for k, v in d["artifacts"].items()},
            updated_at=d["updated_at"],
        )

    @classmethod
    def from_source(cls, path: Path):
        d = json.load(Path(path).open("r", encoding="utf-8"))
        return cls.from_dict(d)
