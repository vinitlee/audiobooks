from typing import List, Dict
import json
from pathlib import Path
import soundfile as sf
import logging
import shutil

from typing import Iterable


def write_json_atomic(d: Dict, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(d, indent=4), encoding="utf-8")
    tmp_path.replace(path)


def write_jsonl_atomic(l: List[Dict], path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for obj in l:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_sound_atomic(path: Path, data: Iterable, samplerate: int, **kwargs):
    tmp_path = path.with_suffix(".tmp" + path.suffix)
    sf.write(file=tmp_path, data=data, samplerate=samplerate, **kwargs)
    tmp_path.replace(path)


def copy_atomic(src: Path, dest: Path):
    tmp_dest = dest.with_suffix(".tmp" + dest.suffix)
    shutil.copy(src, tmp_dest)
    tmp_dest.replace(dest)
