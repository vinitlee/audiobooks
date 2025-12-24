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
    Pattern,
)

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken

import warnings
from pathlib import Path

import yaml
import re

# from torch.cuda import is_available as is_cuda_available
from functools import lru_cache

from dataclasses import dataclass

# %%


class SubRule:
    __slots__ = ("pattern", "replace")

    pattern: Pattern[str]
    replace: str

    def __init__(self, pattern: str, replace: str):
        self.pattern = re.compile(pattern)
        self.replace = replace

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text)


class Lexicon:
    _valid_suffixes = [".yaml"]

    g2g_paths: List[str]
    g2p_paths: List[str]

    g2g_rules: List[SubRule]
    g2p_map: Dict[str, str]

    def __init__(
        self,
        g2g_paths: Optional[List[str]] = None,
        g2p_paths: Optional[List[str]] = None,
    ) -> None:
        self.g2g_paths = []
        self.g2p_paths = []

        self.g2g_rules = []
        self.g2p_map = {}

        self.set_maps(g2g_paths, g2p_paths)

    def set_maps(
        self,
        g2g_paths: Optional[List[str]] = None,
        g2p_paths: Optional[List[str]] = None,
    ):
        self.g2g_paths = []
        self.g2p_paths = []

        self.g2g_rules = []
        self.g2p_map = {}
        self.extend_maps(g2g_paths, g2p_paths)

    def extend_maps(
        self,
        g2g_paths: Optional[List[str]] = None,
        g2p_paths: Optional[List[str]] = None,
    ):
        # Add to current (may be blank)
        self.g2g_paths.extend(g2g_paths or [])
        self.g2p_paths.extend(g2p_paths or [])

        self.g2g_paths = self.filter_valid_paths(self.g2g_paths)
        self.g2p_paths = self.filter_valid_paths(self.g2p_paths)

        self.g2g_from_src(self.g2g_paths)
        self.g2p_from_yaml(self.g2p_paths)

    def filter_valid_paths(self, paths: List[str]) -> List[str]:
        p_paths = [Path(p).expanduser().resolve() for p in paths if p]
        p_paths = [
            p for p in p_paths if p.is_file() and p.suffix in self._valid_suffixes
        ]
        unique_paths = [str(p) for p in p_paths]
        unique_paths = list(dict.fromkeys(unique_paths))
        return unique_paths

    def map_from_yaml(self, paths: List[str]) -> Dict[str, str]:
        mapping = {}
        for p in paths:
            file = Path(p)
            if file.suffix in self._valid_suffixes:
                loaded_map = yaml.safe_load(file.open("r", encoding="utf-8")) or {}
                if loaded_map and isinstance(loaded_map, dict):
                    mapping |= loaded_map
                else:
                    warnings.warn(f"{p} contains non-mapping data.")
        return {str(k): "" if v is None else str(v) for k, v in mapping.items()}

    def g2g_from_src(self, paths: list[str]):
        self.g2g_rules = []
        for p in paths:
            lines = Path(p).read_text(encoding="utf-8").splitlines()
            for ln, l in enumerate(lines, start=1):
                s = l.strip()
                if not s or s.startswith("#"):
                    continue
                y = yaml.safe_load(l)
                if isinstance(y, dict) and len(y):
                    pat, rep = list(y.items())[0]
                    # Strict enforcement
                    if not (isinstance(pat, str) and isinstance(rep, str)):
                        warnings.warn(
                            f"G2G Lexicons must be str:str. Issue in {p} @ {ln}"
                        )
                        continue
                    # Uncomment to enable casting from other types
                    # pat = str(pat)
                    # if rep is None:
                    #     rep = ""
                    # rep = str(rep)
                    self.g2g_rules.append(SubRule(pat, rep))
                else:
                    warnings.warn(f"Found non-mapping data. Skipping {p}")

    def g2p_from_yaml(self, paths: list[str]):
        mapping = self.map_from_yaml(paths)
        mapping = {k.lower(): v for k, v in mapping.items()}
        self.g2p_map = mapping

    def g2g(self, string: str) -> str:
        for rule in self.g2g_rules:
            string = rule(string)
        return string

    def in_g2p(self, grapheme: str) -> bool:
        return grapheme.lower() in self.g2p_map

    def g2p(self, grapheme: str, default_p: str = "") -> str:
        # TODO: Consider adding a fallback function that logs uncommon words not in g2p to class variable
        return self.g2p_map.get(grapheme.lower(), default_p)

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
    def instance(cls, *args, **kwargs) -> KPipelineLazy:
        return cls(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)


class TTSProcessor:
    pipeline: KPipelineLazy
    lexicon: Lexicon
    _unknown_words: dict[str, str] = {}

    def __init__(
        self,
        lexicon: Lexicon,
        pipeline_args: dict = {},
    ) -> None:
        self._unknown_words = {}
        self.init_pipeline(pipeline_args)

    def init_pipeline(self, pipeline_args):
        defaults = {
            "lang_code": "a",
            "device": "cuda",
            "repo_id": "hexgrad/Kokoro-82M",
        }
        # if not is_cuda_available():
        #     defaults["device"] = "cpu"

        self.pipeline = KPipelineLazy.instance(**(defaults | pipeline_args))

    def get_unknown_words(self):
        """
        Use to generate an unknown words document
        """
        return self._unknown_words

    def replacement_g2p(
        self, stock_g2p: Callable[[str], Tuple[str, List[MToken]]]
    ) -> Callable[[str], Tuple[str, List[MToken]]]:
        from misaki import en, espeak

        espeak_fallback = espeak.EspeakFallback(british=False)

        def logging_fallback(t: MToken):

            espeak_token = espeak_fallback(t)

            text_lower = t.text.strip().lower()
            if not self.lexicon.in_g2p(t.text):
                if not text_lower in self._unknown_words:
                    self._unknown_words[text_lower] = espeak_token[0] or ""

            return espeak_token

        logging_g2p = en.G2P(trf=False, british=False, fallback=logging_fallback)

        def replacement_fn(text: str) -> Tuple[str, List[MToken]]:
            text = self.lexicon.g2g(text)
            # TODO: make sure this is in the right place.
            # I am not sure what text looks like here, but it might be too small.
            # Try to run g2g on full chapter text.

            gs, tokens = logging_g2p(text)

            for t in tokens:
                gr = t.text
                ph = t.phonemes

                if self.lexicon.in_g2p(gr):
                    t.text = self.lexicon.g2p(gr, ph or "")

            return gs, tokens

        return replacement_fn
