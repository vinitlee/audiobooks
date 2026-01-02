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
    ClassVar,
    Generator,
)

from numpy._typing import NDArray

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken

from pathlib import Path

import re

import random
import numpy as np
import soundfile as sf
import resampy

from dataclasses import dataclass, field
from functools import lru_cache

from .pipeline import KPipelineLazy, DEFAULT_KOKORO_REPO
from .lexicon import Lexicon
from audiobooks.core import Chapter, ElementBlock

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


class BlockList:
    raw_elements: list[ElementBlock]

    def __init__(self, chapter: Chapter) -> None:
        self.raw_elements = chapter.elements

    @property
    def blocks(self):
        return list(self.converted())

    @property
    def strings(self):
        return [str(b) for b in self.blocks]

    def converted(self):
        yield from self.head()
        for el in self.raw_elements:
            yield from self.convert_element(el)
        yield from self.foot()

    def head(self) -> Iterable[Block]:
        return ()

    def foot(self) -> Iterable[Block]:
        return (PauseBlock(0.5),)

    def convert_element(self, el: ElementBlock) -> Iterable[Block]:
        match el.tag:
            case "p":
                yield from self.conv_text(el)
            case "h1" | "h2" | "h3":
                yield from self.conv_title(el)
            case _:
                yield from self.conv_default(el)

    @staticmethod
    def conv_title(el: ElementBlock) -> Iterable[Block]:
        transition_sounds = [
            # r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#2-1767349191268.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#2-1767349479476.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#3-1767349197004.wav",
            # r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#3-1767349479477.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#4-1767349197004.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#4-1767349487169.wav",
        ]

        return (
            # WavBlock.from_file(random.choice(transition_sounds)),
            PauseBlock(0.2),
            TextBlock(el.text),
            PauseBlock(0.5),
        )

    @staticmethod
    def conv_text(el: ElementBlock) -> Iterable[Block]:
        return (TextBlock(el.text),)

    @staticmethod
    def conv_default(el: ElementBlock) -> Iterable[Block]:
        return (Block(),)


@dataclass
class Block:
    override: ClassVar[bool] = False

    def audio_data(self, sr: int) -> np.typing.NDArray:
        return np.array([], dtype=np.float32)

    def __str__(self) -> str:
        return ""


@dataclass
class TextBlock(Block):
    text: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass
class AudioBlock(Block):
    override: ClassVar[bool] = True

    def audio_data(self, sr: int) -> np.typing.NDArray:
        return np.array([])

    def __str__(self) -> str:
        return ""


@dataclass
class WavBlock(AudioBlock):
    data: np.typing.NDArray
    _resample_cache: dict = field(default_factory=dict, init=False, repr=False)
    sr: int

    @classmethod
    def from_file(cls, path: Path):
        file_data, file_sr = sf.read(path, dtype="float32")
        assert file_data.dtype == np.float32
        if np.ndim(file_data) > 1 and file_data.shape[1] > 1:
            file_data = np.mean(file_data, axis=1)
        return cls(file_data, file_sr)

    def resample(self, sr_new: int):
        if sr_new not in self._resample_cache:
            self._resample_cache[sr_new] = resampy.resample(self.data, self.sr, sr_new)

        return self._resample_cache[sr_new]

    def audio_data(self, sr: int) -> np.typing.NDArray:
        if sr != self.sr:
            return self.resample(sr)
        return self.data


@dataclass
class RandomAudioBlock(AudioBlock):
    possible_blocks: Iterable[AudioBlock]

    def audio_data(self, sr: int) -> np.ndarray[Tuple[Any], np.dtype]:
        selection_idx = np.random.randint(0, len(self.possible_blocks) - 1)
        return self.possible_blocks[selection_idx].audio_data(sr)


@dataclass
class PauseBlock(AudioBlock):
    pause_length: float = 0

    def audio_data(self, sr: int) -> np.typing.NDArray:
        if self.pause_length == 0:
            return np.array([])
        return np.zeros(int(self.pause_length * sr), dtype=np.float32)


# %%
