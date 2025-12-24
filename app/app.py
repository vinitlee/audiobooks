from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

from typing import Any, Callable, Iterable, Optional, Set, Union, List, Dict, Union

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel

import yaml

import argparse
import glob

from project import AudiobookProject

from pathlib import Path


def main(
    paths: Set[str],
    voice: str = "am_michael",
    speed: float = 1,
    lexicon_g2g: List[str] | None = None,
    lexicon_g2p: List[str] | None = None,
    metadata_overrides=None,
    output: str | None = None,
    init_only=False,
    clean=False,
):

    for path in paths:
        params = clean_dict(
            {
                "init_path": path,
                "tts_voice": voice,
                "tts_speed": speed,
                "lex_g2g_paths": lexicon_g2g,
                "lex_g2p_paths": lexicon_g2p,
                "override_metadata": metadata_overrides,
            }
        )
        proj = AudiobookProject(**params)

        if init_only:
            continue

        proj.make_splits()
        proj.make_master()
        proj.make_m4b()
        proj.copy_to_library()


def not_none_dict(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def clean_dict(d: dict):
    return {k: v for k, v in d.items() if v}


def ensure_list(e):
    if not isinstance(e, list):
        return [e]
    return e


def parse_config(config_path):
    config = yaml.safe_load(Path(config_path).open())
    config_name = config.get("name")
    config_metadata = clean_dict(
        {
            "author": config.get("author"),
            "series": config.get("series"),
        }
    )
    config_args = clean_dict(
        {
            "paths": config.get("paths") or config.get("files"),
            "voice": config.get("voice"),
            "speed": float(config.get("speed")),
            "lexicon_g2g": ensure_list(config.get("g2g")),
            "lexicon_g2p": ensure_list(config.get("g2p") or config.get("lexicon")),
            "metadata_overrides": config_metadata,
            "output": config.get("output"),
        }
    )

    return config_args


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "paths",
        metavar="PATH",
        nargs="*",
        help="Path(s) to *.epub or dir to be processed; accepts globs",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        help="Kokoro TTS voice",
        choices=[
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
        ],
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        help="Kokoro TTS Speed",
    )
    parser.add_argument(
        "--g2g",
        "-lg",
        type=str,
        help="Path to external G2G lexicon YAML file",
    )
    parser.add_argument(
        "--g2p",
        "-lp",
        type=str,
        help="Path to external G2P lexicon YAML file",
    )
    parser.add_argument(
        "--init-only",
        "-ino",
        action="store_true",
        help="Just set up project, don't generate",
    )
    parser.add_argument(
        "--clean",
        "-cc",
        action="store_true",
        help="Delete project files after running. Ignored if output not set.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Directory to copy final M4B when done",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Override title for all files",
    )
    parser.add_argument(
        "--author",
        type=str,
        help="Override author for all files",
    )
    parser.add_argument(
        "--series",
        type=str,
        help="Override series for all files",
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="PATH",
        help="Specify YAML config. YAML will be overridden by any cmd arguments.",
    )
    args = parser.parse_args()

    set_args = {}

    if args.config:
        set_args = parse_config(args.config)
    cmd_args = clean_dict(
        {
            "paths": args.paths,
            "voice": args.voice,
            "speed": args.speed,
            "lexicon_g2g": args.g2g,
            "lexicon_g2p": args.g2p,
            "output": args.output,
        }
    )
    cmd_metadata_overrides = clean_dict(
        {
            "title": args.title,
            "author": args.author,
            "series": args.series,
        }
    )
    cmd_only_args = clean_dict(
        {
            "clean": args.clean,
            "init_only": args.init_only,
        }
    )

    print(set_args)
    print(cmd_args)
    print(cmd_metadata_overrides)
    print(cmd_only_args)

    set_args.update(cmd_args)
    if not isinstance(set_args.get("metadata_overrides"), dict):
        set_args["metadata_overrides"] = {}
    set_args["metadata_overrides"].update(cmd_metadata_overrides)
    set_args.update(cmd_only_args)

    if set_args.get("speed"):
        set_args["speed"]

    expanded_paths = set()
    for p in set_args["paths"]:
        p = str(Path(p).expanduser().resolve())
        expanded_paths = expanded_paths.union(glob.glob(p))
    set_args["paths"] = sorted(list(expanded_paths))

    print(f"\n------ args ------\n{set_args}\n------------------\n")
    return set_args


my_proj = None
if __name__ == "__main__":
    arguments = parse_arguments()

    main(**arguments)
