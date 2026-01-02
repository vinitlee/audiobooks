import json
from pathlib import Path
import soundfile as sf

from typing import Iterable


def write_json_atomic(d: dict, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(d, indent=4), encoding="utf-8")
    tmp_path.replace(path)


def write_sound_atomic(path: Path, data: Iterable, samplerate: int, **kwargs):
    tmp_path = path.with_suffix(".tmp" + path.suffix)
    sf.write(file=tmp_path, data=data, samplerate=samplerate, **kwargs)
    tmp_path.replace(path)
