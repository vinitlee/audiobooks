# %%
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
import dpath.util

import argparse
import yaml

import numpy as np
import cv2

import soundfile as sf

from ebooklib import epub
import ebooklib

import warnings
from tqdm import tqdm
import traceback

from book import Book

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

        # Make Book
        self.set_state_book_metadata(override_metadata)
        self.init_book()

        # Make Lexicon
        self.init_lexicon(lex_g2g_paths, lex_g2p_paths)
        self.set_state_tts(tts_voice, tts_speed)

        # Mark init completed and save progress
        self.complete["init"] = True
        self.save_state()

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
        print(self.state)
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


class Lexicon:
    g2g_map: dict[str, str] = {}
    g2p_map: dict[str, str] = {}
    g2g_paths: list[str] = []
    g2p_paths: list[str] = []
    _valid_suffixes = [".yaml"]

    def __init__(
        self,
        g2g_paths: list[str] = [],
        g2p_paths: list[str] = [],
    ) -> None:
        self.set_maps(g2g_paths, g2p_paths)

    def set_maps(
        self,
        g2g_paths: list[str] = [],
        g2p_paths: list[str] = [],
    ):
        self.g2g_paths = []
        self.g2p_paths = []
        self.g2g_map = {}
        self.g2p_map = {}
        self.extend_maps(g2g_paths, g2p_paths)

    def extend_maps(
        self,
        g2g_paths: list[str] = [],
        g2p_paths: list[str] = [],
    ):
        # Add to current (may be blank)
        self.g2g_paths.extend(set(g2g_paths))
        self.g2p_paths.extend(set(g2p_paths))

        self.g2g_paths = self.filter_valid_paths(self.g2g_paths)
        self.g2p_paths = self.filter_valid_paths(self.g2p_paths)

        self.g2g_map = self.map_from_yaml(self.g2g_paths)
        self.g2p_map = self.map_from_yaml(self.g2p_paths)

    def filter_valid_paths(self, paths: list[str]) -> list[str]:
        p_paths = [Path(p).expanduser().resolve() for p in paths if p]
        p_paths = [
            p for p in p_paths if p.is_file() and p.suffix in self._valid_suffixes
        ]
        unique_paths = list(set([str(p) for p in p_paths]))
        return unique_paths

    def map_from_yaml(self, paths: list[str]):
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


class AudioProcessor:
    def __init__(self, splits: list[AudioChapter] = []):
        pass

    def add_split(self):
        pass

    # needs more shape


class TTSProcessor:
    pipeline: KPipelineLazy

    def __init__(
        self,
        lexicon: Lexicon,
        pipeline_external: Optional[KPipelineLazy] = None,
        pipeline_args: dict = {},
    ) -> None:
        if pipeline_external is not None:
            self.pipeline = pipeline_external
        else:
            self.init_pipeline(pipeline_args)

    def init_pipeline(self, pipeline_args):
        global PIPELINE
        try:
            if not isinstance(PIPELINE, KPipelineLazy):  # type: ignore
                raise Exception("PIPELINE unbound")
        except:
            defaults = {
                "lang_code": "a",
                "device": "cuda",
                "repo_id": "hexgrad/Kokoro-82M",
            }
            PIPELINE = KPipelineLazy(**(defaults | pipeline_args))
            self.pipeline = PIPELINE

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
