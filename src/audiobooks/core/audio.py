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

import soundfile as sf

# %%


class AudioChapter:
    title: Optional[str] = None
    audio_file: Optional[Path | str] = None
    duration: Optional[int] = None
    hash: Optional[int] = None  # hash of input text + TTS params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "audio_file": str(self.audio_file),
            "duration": self.duration,
        }


class AudioProcessor:
    # TODO: Plan and implement

    def __init__(self, splits: list[AudioChapter] = []):
        pass

    def add_split(self):
        pass
