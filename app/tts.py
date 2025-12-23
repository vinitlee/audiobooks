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

import warnings
from pathlib import Path

import yaml
import re

import torch
from functools import lru_cache

# %%


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

    def g2g_from_yaml(self, paths: list[str]):
        return self.map_from_yaml(paths)

    def g2p_form_yaml(self, paths: list[str]):
        mapping = self.map_from_yaml(paths)
        mapping = {k.lower(): v for k, v in mapping.items()}
        return mapping

    def g2g(self, string: str) -> str:
        for k, v in self.g2g_map.items():
            string = re.sub(k, v, string)
        return string

    def in_g2p(self, grapheme: str) -> bool:
        return grapheme.lower() in self.g2p_map

    def g2p(self, grapheme: str, default_p: str = "") -> str:
        # TODO: Consider adding a fallback function that logs uncommon words not in g2p to class variable
        return self.g2p_map.get(grapheme.lower(), default_p)

    def kp_lex_g2p(
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
                    g = t.text.lower()
                    if g in self.g2g_map:
                        t.phonemes = self.g2p(g, t.phonemes or "")

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

    @classmethod
    @lru_cache(maxsize=1)
    def instance(cls) -> KPipelineLazy:
        return cls()

    def __getattr__(self, name: str):
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)


class TTSProcessor:
    # TODO: Plan and implement

    pipeline: KPipelineLazy
    lexicon: Lexicon
    unknown_words: dict[str, str] = {}

    def __init__(
        self,
        lexicon: Lexicon,
        pipeline_args: dict = {},
    ) -> None:
        self.init_pipeline(pipeline_args)

    def init_pipeline(self, pipeline_args):
        defaults = {
            "lang_code": "a",
            "device": "cuda",
            "repo_id": "hexgrad/Kokoro-82M",
        }
        if not torch.cuda.is_available():
            defaults["device"] = "cpu"

        self.pipeline = KPipelineLazy.instance(**(defaults | pipeline_args))

    def dump_unknown_words(self):
        pass

    def logging_fallback(self):
        from misaki import espeak

        self._espeak = espeak.EspeakFallback(british=False)

        def fallback_fn(t: MToken):
            ef = self._espeak(t)

            text_lower = t.text.strip().lower()
            if not self.lexicon.in_g2p(t.text):
                if not text_lower in self.unknown_words:
                    self.unknown_words[text_lower] = ef[0] or ""

            return ef

        return fallback_fn

    def replacement_g2p(
        self, stock_g2p: Callable[[str], Tuple[str, List[MToken]]]
    ) -> Callable[[str], Tuple[str, List[MToken]]]:
        from misaki import en

        logging_g2p = en.G2P(trf=False, british=False, fallback=self.logging_fallback())

        def replacement_fn(text: str) -> Tuple[str, List[MToken]]:
            text = self.lexicon.g2g(text)
            # TODO: it's possible this should actually just be rewritten into g2p doing regex
            # Seems like it might still be good to have both

            gs, tokens = logging_g2p(text)

            for t in tokens:
                gr = t.text
                ph = t.phonemes

                if self.lexicon.in_g2p(gr):
                    t.text = self.lexicon.g2p(gr, ph or "")

            return gs, tokens

        return replacement_fn
