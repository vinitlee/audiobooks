from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel

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
    Literal,
    ClassVar,
    Generator,
    Protocol,
    Mapping,
)
from numpy._typing import NDArray

from misaki.en import G2P, MToken
from misaki import en, espeak

from pathlib import Path

import re

import random
import numpy as np
import soundfile as sf
import resampy
import wordfreq

from dataclasses import dataclass, field
from functools import lru_cache

from .pipeline import KPipelineLazy, DEFAULT_KOKORO_REPO
from .lexicon import Lexicon

# from audiobooks.core import Chapter, ElementBlock
from audiobooks.tts.lexicon import LexicalMap, LexicalPatternMap
from .blocks import *

from tqdm import tqdm
import logging

# %%
Voice = Literal[
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "bf_emma",
    "bf_isabella",
    "am_adam",
    "am_michael",
    "bm_george",
    "bm_lewis",
]

ComputeDevice = Literal["cpu", "cuda"]
LanguageCode = Literal["a", "b", "e", "f", "h", "i", "j", "p", "z"]


class TTSProcessor:
    pipeline: KPipelineLazy
    lexicon: Lexicon
    sample_rate: ClassVar[int] = 24000
    voice: Voice
    speed: float
    _saved_g2p: G2P
    replacement_g2p: PatchedMisakiG2P

    def __init__(
        self,
        lexicon: Lexicon,
        voice: Voice = "am_michael",
        speed: float = 1.0,
        pipeline_args: dict = {},
    ) -> None:
        self.voice = voice
        self.speed = speed

        self.replacement_g2p = PatchedMisakiG2P(lexicon.g2g, lexicon.g2p)

        self.init_pipeline(**pipeline_args)

    def init_pipeline(
        self,
        lang_code: LanguageCode = "a",
        device: ComputeDevice = "cuda",
        repo_id: str = DEFAULT_KOKORO_REPO,
    ):
        args = {
            "lang_code": lang_code,
            "device": device,
            "repo_id": repo_id,
        }

        self.pipeline = KPipelineLazy.instance(**args)

    def generate(self, bl: BlockList, progress_title: str = "TTS Generate"):

        blocks = bl.blocks
        strings = bl.strings

        self.pipeline.load()
        if self.pipeline._obj is None:
            raise RuntimeError("Pipeline object is None")

        self.patch_pipeline()

        generator = self.pipeline(
            strings,
            voice=self.voice,
            speed=self.speed,
        )

        audio_clips: list[list[np.typing.NDArray]] = [[] for b in blocks]

        # FIXME: This count is wrong and it's messy
        word_counts = [len(s.split()) for s in strings]
        total_words = sum(word_counts)
        word_counts_iter = iter(word_counts)
        progress = tqdm(total=total_words, desc=progress_title, unit="words")

        for result in generator:
            idx = result.text_index
            audio_tensor = result.audio
            logging.debug(str(idx), str(audio_tensor))
            if idx is not None and audio_tensor is not None:
                audio = audio_tensor.detach().cpu().numpy()
                audio_clips[idx].append(audio)
            progress.update(next(word_counts_iter, 0))
        progress.close()

        for i, b in enumerate(blocks):
            if b.override:
                audio_clips[i] = [b.audio_data(self.sample_rate)]

        audio_parts = [np.concat(c) for c in audio_clips]

        self.restore_pipeline()

        return audio_parts

    def patch_pipeline(self):
        if self.pipeline._obj is None:
            raise RuntimeError("Pipeline object is None")

        if isinstance(self.pipeline._obj.g2p, G2P):
            self._saved_g2p = self.pipeline._obj.g2p
        else:
            raise RuntimeError(
                f"G2P was of a different type: {type(self.pipeline._obj.g2p).__name__}"
                "This project only support English G2P."
            )

        self.pipeline._obj.g2p = self.replacement_g2p

    def restore_pipeline(self):
        if self.pipeline._obj is None:
            raise RuntimeError("Pipeline object is None")

        self.pipeline.g2p = self._saved_g2p

    @property
    def unknown_words(self):
        """
        Use to generate an unknown words document
        """
        return self.replacement_g2p.unknown_words


class PatchedMisakiLexicon(en.Lexicon):
    patch_rating: int = 5
    patch_g2p_map: LexicalMap

    def __init__(self, british, g2p_lut: LexicalMap):
        super().__init__(british=british)
        self.patch_g2p_map = g2p_lut

    def lookup_(self, tk: MToken) -> str | None:
        return self.patch_g2p_map(tk.text)

    @property
    def keys(self) -> List[str]:
        return list(self.patch_g2p_map.keys())

    def __call__(self, tk, ctx):
        if (result := self.lookup_(tk)) is not None:
            return (result, self.patch_rating)
        return super().__call__(tk, ctx)


PreprocessRtn = Tuple[str, list, dict]
PreprocessFn = Callable[[str], PreprocessRtn]
PreprocessArg = Union[PreprocessFn, bool]


class PatchedMisakiG2P(G2P):
    _espeak: espeak.EspeakFallback
    rare_word_threshold: ClassVar[float] = 1e-7

    g2g_pattern_map: LexicalPatternMap

    unknown_words: Dict[str, Optional[str]]

    def __init__(
        self, g2g_pattern_map: LexicalPatternMap, g2p_map: LexicalMap, **kwargs
    ):
        """Initialize the G2P engine with a spaCy pipeline, lexicon configuration, and optional fallback resolver."""
        super().__init__(**kwargs)

        british = kwargs.get("british", False)

        # G2G
        self.g2g_pattern_map = g2g_pattern_map

        # G2P
        self.lexicon = PatchedMisakiLexicon(british, g2p_map)

        self._espeak = espeak.EspeakFallback(british=british)
        self.fallback = self.fallback_

        self.unknown_words = {}

    def fallback_(self, token: MToken) -> tuple[str | None, int | None]:
        espeak_output = self._espeak(token)

        # Check if in lexicon (it generally should not be)
        if token.text not in self.lexicon.keys:
            # If not, check English frequency and log to lexicon if low
            if wordfreq.word_frequency(token.text, "en") < self.rare_word_threshold:
                self.unknown_words[token.text] = espeak_output[0]

        return espeak_output

    def __call__(
        self,
        text: str,
        preprocess: PreprocessArg = True,
    ) -> Tuple[str, List[MToken]]:
        if preprocess is False:
            raise ValueError("preprocess set to False instead of True or a Callable")

        def _preprocess(text: str) -> PreprocessRtn:
            text = self.g2g_pattern_map(text)
            if preprocess is True:
                return G2P.preprocess(text)
            else:
                return preprocess(text)

        return super().__call__(text, preprocess=_preprocess)  # type: ignore


# %%
