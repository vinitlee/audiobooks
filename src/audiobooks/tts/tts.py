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
    Literal,
)

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken

from pathlib import Path

import re

from dataclasses import dataclass

from .pipeline import KPipelineLazy, DEFAULT_KOKORO_REPO
from .lexicon import Lexicon

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
    _unknown_words: dict[str, str] = {}

    def __init__(
        self,
        lexicon: Lexicon,
        pipeline_args: dict = {},
    ) -> None:
        self._unknown_words = {}
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
