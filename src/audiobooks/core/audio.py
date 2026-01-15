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

import ffmpeg

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


def master_audio_from_chapters(input_paths: list[Path], output_path: Path, **kwargs):
    input_streams = [ffmpeg.input(str(p)) for p in input_paths[:2]]

    output_args = {
        "c:a": "aac",
        "aac_coder": "fast",
        "b:a": "48k",
        "ac": 1,
        "ar": 24000,
    }
    output_args |= kwargs

    graph = (
        ffmpeg.concat(*input_streams, v=0, a=1)
        .filter("loudnorm", I=-16, TP=-1.5, LRA=11)
        .output(str(output_path), **output_args)
        .overwrite_output()
    )

    graph.run()


def m4b_from_master_audio(
    audio_path: Path,
    cover_path: Path,
    meta_path: Path,
    output_path: Path,
    **kwargs,
):

    audio_input = ffmpeg.input(str(audio_path))
    cover_input = ffmpeg.input(str(cover_path))

    a = audio_input["a"]
    v = cover_input["v"]
    m = ffmpeg.input(str(meta_path))

    output_args: dict = {
        "c:a": "copy",
        "c:v": "copy",
        "map_metadata": 2,
        "map_chapters": 2,
        "metadata:s:v:0": 'title="Cover"',
        "metadata:s:v:0": 'comment="Cover (front)"',
        "disposition:v:0": "attached_pic",
        "movflags": "+faststart",
    }

    graph = ffmpeg.output(a, v, m, str(output_path), **output_args)

    graph.run()


class AudioProcessor:
    # TODO: Plan and implement

    def __init__(self, splits: list[AudioChapter] = []):
        pass

    def add_split(self):
        pass
