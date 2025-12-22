# %%
from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken
    from typing import Any, Callable, Iterable, Optional, Tuple, List, Dict

from line_profiler import profile
from pathlib import Path
import glob
from pathvalidate import sanitize_filepath
import json
import re
import subprocess
import datetime
import shutil
from dataclasses import dataclass

import argparse
import yaml

import numpy as np
import cv2

import soundfile as sf

from ebooklib import epub
import ebooklib

import warnings
from tqdm import tqdm

from book import Book

# %%


class AudiobookProject:
    book: Book
    lexicon: Lexicon
    override_metadata: Dict[str, Any] = {}
    # Generation
    processor: TTSProcessor
    tts_voice: str
    tts_speed: float
    # Paths
    dir: Path
    epub: str
    projdata: str = "project.yaml"
    fulltext: str = "fulltext.txt"
    cover: str = "cover.jpg"
    wavfiles: str = "files"
    ffmeta: str = "ffmetadata"

    complete: dict[str, bool] = {
        "init": False,
        "chapters": False,
        "join": False,
        "postprocess": False,
        "m4b": False,
        "copy": False,
    }

    chapters: list[AudioChapter] = []

    def __init__(
        self,
        init_path: Path | str,
        lexicon: Optional[Lexicon] = None,
        override_metadata: Dict[str, Any] = {},
    ) -> None:

        init_path = Path(init_path)

        self.override_metadata = override_metadata

        initialized = False
        if init_path.is_dir():
            self.dir = init_path
            try:
                # If there is a project.yaml
                self.load()
                initialized = True
            except:
                initialized = False
            if not initialized and len(list(self.dir.glob("*.epub"))):
                # Fallback if loading project.yaml fails; it will be overwritten
                self.lexicon = lexicon or Lexicon()
                self.epub = next(self.dir.glob("*.epub")).name
                self.init_from_epub()
                initialized = True
        elif init_path.exists() and init_path.suffix == ".epub":
            # init_path is epub.

            # Make project dir
            self.dir = init_path.parent / init_path.stem
            self.dir = self.dir.resolve()
            self.dir.mkdir(parents=True, exist_ok=False)

            # Move epub into it
            self.epub = init_path.name
            init_path.rename(self.dir / self.epub)

            self.lexicon = lexicon or Lexicon()
            self.init_from_epub()
            initialized = True
        if not initialized:
            raise Exception(f"Failed to initialized at {init_path}")

    def init_book(self):
        self.book = Book(self.dir / self.epub)
        self.book.meta._overrides = self.override_metadata

    def init_from_epub(self):
        self.init_book()
        self.dump()

    def load(self):
        yaml_path = self.dir / self.projdata

        struct = yaml.safe_load(yaml_path.open("r", encoding="utf-8"))

        self.dir = Path(struct["Project Directory"])
        self.epub = struct["EPUB"]
        self.init_book()
        self.complete = struct.get("Progress")
        self.chapters = [AudioChapter(**c) for c in struct.get("Chapters", {})]
        self.lexicon = Lexicon(struct["Lexicon"]["g2g"], struct["Lexicon"]["g2p"])

    def dump(self):
        yaml_path = self.dir / self.projdata

        struct = {
            "Title": self.book.meta.title,
            "Project Directory": str(self.dir),
            "EPUB": str(self.epub),
            "Progress": self.complete,
            "Chapters": [c.to_dict() for c in self.chapters],
            "Lexicon": self.lexicon.to_dict(),
        }

        yaml.dump(
            struct,
            yaml_path.open("w", encoding="utf-8"),
            allow_unicode=True,
            sort_keys=False,
        )


class Lexicon:
    g2g_map: dict[str, str] = {}
    g2p_map: dict[str, str] = {}
    g2g_paths: list[Path | str] = []
    g2p_paths: list[Path | str] = []
    _valid_suffixes = [".yaml"]

    def __init__(
        self,
        g2g_paths: list[Path | str] = [],
        g2p_paths: list[Path | str] = [],
    ) -> None:
        self.set_maps(g2g_paths, g2p_paths)

    def set_maps(
        self,
        g2g_paths: list[Path | str] = [],
        g2p_paths: list[Path | str] = [],
    ):
        self.g2g_paths = []
        self.g2p_paths = []
        self.g2g_map = {}
        self.g2p_map = {}
        self.extend_maps(g2g_paths, g2p_paths)

    def extend_maps(
        self,
        g2g_paths: list[Path | str] = [],
        g2p_paths: list[Path | str] = [],
    ):
        self.g2g_paths.extend(g2g_paths)
        self.g2p_paths.extend(g2p_paths)
        self.g2g_map = self.map_from_yaml(g2g_paths)
        self.g2p_map = self.map_from_yaml(g2p_paths)

    def map_from_yaml(self, paths: list[Path | str]):
        mapping = {}
        for p in paths:
            file = Path(p)
            if file.suffix in self._valid_suffixes:
                mapping |= yaml.safe_load(file.open("r", encoding="utf-8"))
        return mapping

    def g2g(self, string: str) -> str:
        for k, v in self.g2g_map.items():
            string = re.sub(k, v, string)
        return string

    def g2p(
        self, stock_g2p: Callable[[str], tuple[Iterable, Iterable]]
    ) -> Callable[[str], tuple[Iterable, Iterable]]:
        def g2p_fn(text: str) -> tuple[Iterable, Iterable]:
            gs, tokens = stock_g2p(text)

            if tokens is not None:
                for t in tokens:
                    k = t.text.lower()
                    if k in self.g2g_map:
                        t.phonemes = self.g2g_map[k]

            return gs, tokens

        return g2p_fn

    # TODO: Consider adding a fallback function that logs uncommon words not in g2p to class variable

    def g2g2p(
        self, stock_g2p: Callable[[str], Tuple[str, List[MToken]]]
    ) -> Callable[[str], Tuple[str, List[MToken]]]:
        """
        Closure that modifies an input g2p function from your KPipeline
        to first apply g2g lexicon mappings, run the original g2p,
        and then apply corrections with the g2p mappings per-token
        """

        def g2g2p_fn(text: str) -> Tuple[str, List[MToken]]:
            text = self.g2g(text)

            gs, tokens = stock_g2p(text)

            if tokens is not None:
                for t in tokens:
                    k = t.text.lower()
                    if k in self.g2g_map:
                        t.phonemes = self.g2g_map[k]

            return gs, tokens

        return g2g2p_fn

    def to_dict(self):
        return {
            "g2g": self.g2g_paths,
            "g2p": self.g2p_paths,
        }


class KPipelineLazy:
    g2p: Callable[[str], Tuple[str, List[MToken]]]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._obj = None

    def load(self):
        if self._obj is None:
            from kokoro import KPipeline

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                self._obj = KPipeline(*self.args, **self.kwargs)
        return self._obj

    def __getattr__(self, name: str) -> profile:
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)


@dataclass
class AudioChapter:
    title: Optional[str] = None
    audio_file: Optional[Path | str] = None
    duration: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "audio_file": str(self.audio_file),
            "duration": self.duration,
        }


class TTSProcessor:
    pipeline: KPipelineLazy

    def __init__(self, pipeline_args={}) -> None:
        self.init_pipeline(pipeline_args)

    def init_pipeline(self, pipeline_args):
        defaults = {"lang_code": "a", "device": "cuda", "repo_id": "hexgrad/Kokoro-82M"}
        self.pipeline = KPipelineLazy(**(defaults | pipeline_args))

    def process(self, segments, output_path, lexicon: Lexicon | None = None):
        if lexicon is None:
            lexicon = Lexicon()

        # Swap out G2P
        base_g2p = self.pipeline.g2p
        self.pipeline.g2p = lexicon.g2g2p(base_g2p)

        # Clean up
        self.pipeline.g2p = base_g2p

    # def generate_sound_files(self, overwrite=False):
    #     # TODO: Make skipping more efficient by reading data rather than wiping it
    #     self.data["chapters"] = []
    #     self.data["generated"] = False

    #     # Swap out G2P
    #     stock_g2p = self.pipeline.g2p
    #     self.pipeline.g2p = self.lexicon_g2p(stock_g2p)

    #     for i, chapter in enumerate(self.book.chapters):

    #         output_path = self.project_dir_path / f"split-{i}.wav"

    #         if (not output_path.exists()) or overwrite:
    #             sr = 24000
    #             split_pattern = r"\n+"
    #             generator = self.pipeline(
    #                 chapter.text,
    #                 voice=self.data["voice"],
    #                 speed=self.data["speed"],
    #                 split_pattern=split_pattern,
    #             )
    #             audio_clips = []

    #             segments = [
    #                 s for s in re.split(split_pattern, chapter.text) if s.strip()
    #             ]
    #             segment_word_counts = [len(seg.split()) for seg in segments]
    #             total_words = sum(segment_word_counts)
    #             word_pbar = tqdm(total=total_words, desc=chapter.title, unit="words")
    #             word_count_iter = iter(segment_word_counts)
    #             for gph, phn, snd in generator:
    #                 if type(snd) in [torch.FloatTensor, torch.Tensor]:
    #                     audio_clips.append(snd.detach().cpu().numpy())
    #                 word_pbar.update(next(word_count_iter, 0))
    #             word_pbar.close()
    #             full_audio = np.concat(audio_clips)  # TODO: join with adustable gap
    #             sf.write(output_path, full_audio, sr)
    #         else:
    #             print(f"Chapter[{i}] output exists, skipping.")
    #             full_audio, sr = sf.read(str(output_path))

    #         self.data["chapters"].append(
    #             {
    #                 "number": i + 1,
    #                 "title": chapter.title,
    #                 "length": int(
    #                     1000 * len(full_audio) / sr,
    #                 ),
    #                 "path": str(output_path.resolve()),
    #             }
    #         )

    #     # Restore original G2P
    #     self.pipeline.g2p = stock_g2p

    #     self.data["generated"] = True
    #     self.write_data_file()
