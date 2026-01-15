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

from dataclasses import dataclass, field
from functools import lru_cache

from .pipeline import KPipelineLazy, DEFAULT_KOKORO_REPO
from .lexicon import Lexicon

from audiobooks.core import Chapter, ElementBlock

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
    _unknown_words: dict[str, str]
    _saved_g2p: G2P

    def __init__(
        self,
        lexicon: Lexicon,
        voice: Voice = "am_michael",
        speed: float = 1.0,
        pipeline_args: dict = {},
    ) -> None:
        self._unknown_words = {}

        self.voice = voice
        self.speed = speed

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

    def generate(self, bl: BlockList, progress_title="TTS Generate"):

        blocks = bl.blocks[:5]  # DEBUG
        strings = bl.strings[:5]  # DEBUG

        self.pipeline.load()
        if self.pipeline._obj is None:
            raise RuntimeError("Pipeline object is None")

        if isinstance(self.pipeline._obj.g2p, G2P):
            self._saved_g2p = self.pipeline._obj.g2p
        else:
            raise RuntimeError(
                f"G2P was of a different type: {type(self.pipeline._obj.g2p).__name__}"
                "This project only support English G2P."
            )
        self.pipeline._obj.g2p = self.replacement_g2p()

        generator = self.pipeline(
            strings,
            voice=self.voice,
            speed=self.speed,
        )

        audio_clips: list[list[np.typing.NDArray]] = [[] for b in blocks]

        word_counts = [len(s.split()) for s in strings]
        total_words = sum(word_counts)
        word_counts_iter = iter(word_counts)
        progress = tqdm(total=total_words, desc=progress_title, unit="words")

        print(strings)

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

        self.pipeline.g2p = self._saved_g2p

        return audio_parts

    def get_unknown_words(self):
        """
        Use to generate an unknown words document
        """
        return self._unknown_words

    # TODO: rewrite as a subclass of misaki.G2P
    def replacement_g2p(self) -> Callable[[str], Tuple[str, List[MToken]]]:
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
            orig_text = text  # DEBUG
            text = self.lexicon.g2g(text)
            logging.debug(f"{orig_text} -> {text}")

            gs, tokens = logging_g2p(text)

            for t in tokens:
                gr = t.text
                ph = t.phonemes

                if self.lexicon.in_g2p(gr):
                    t.text = self.lexicon.g2p(gr, ph or "")

            return gs, tokens

        return replacement_fn


class PatchedLexicon(en.Lexicon):
    patch_rating: int = 5
    patch_g2p_lut: Mapping[str, str]

    def __init__(self, british, g2p_lut: Mapping[str, str]):
        super().__init__(british=british)
        self.patch_g2p_lut = g2p_lut

    def lookup_(self, tk: MToken) -> str | None:
        return self.patch_g2p_lut.get(tk.text, None)

    def __call__(self, tk, ctx):
        if (result := self.lookup_(tk)) is not None:
            return (result, self.patch_rating)
        return super().__call__(tk, ctx)


class PatchedG2P(G2P):
    _espeak: espeak.EspeakFallback

    def __init__(self, lexicon_patch: Callable[[str], str], **kwargs):
        """Initialize the G2P engine with a spaCy pipeline, lexicon configuration, and optional fallback resolver."""
        super().__init__(**kwargs)

        british = kwargs.get("british", False)

        self.lexicon = PatchedLexicon(british, lexicon_patch)

        self._espeak = espeak.EspeakFallback(british=british)
        self.fallback = self.fallback_

    def fallback_(self, token: MToken) -> tuple[str | None, int | None]:
        espeak_output = self._espeak(token)

        # Check if in lexicon
        # If not, check English frequency and log to lexicon if low

        return espeak_output

    @staticmethod
    def preprocess(text):
        """Normalize inline markup, extract token-level features, and return aligned text, tokens, and feature map."""

        # Run g2g regexes str -> str

        return G2P.preprocess(text)

    @staticmethod
    def retokenize(tokens: List[MToken]) -> List[Union[MToken, List[MToken]]]:
        """Split tokens into subtokens, handle punctuation and currency cases, and group into lexical word units."""
        result_tokens = G2P.retokenize(tokens)
        for i, w in reversed(list(enumerate(result_tokens))):
            if not isinstance(w, list):
                if w.phonemes is None:
                    # If in Lexicon
                    # w.phonemes, w.rating = lexicon(w.graphemes),5
                    pass
        return result_tokens


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

        # For use with
        # WavBlock.from_file(random.choice(transition_sounds))
        transition_sounds = [
            # r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#2-1767349191268.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#2-1767349479476.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#3-1767349197004.wav",
            # r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#3-1767349479477.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#4-1767349197004.wav",
            r"G:\Projects\audiobooks\assets\audio\strum_harp_gentle_in_#4-1767349487169.wav",
        ]

        return (
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
