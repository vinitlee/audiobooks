from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

from typing import Any, Callable, Iterable, Optional, Set, Union, List, Dict, Union

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel

import yaml

import argparse
import glob
import logging

from pathlib import Path
from audiobooks import AudiobookProject, ProjectConfig, Voice
from audiobooks.utils import not_none_dict, clean_dict, ensure_list


def main(
    paths: List[str],
    voice: Voice = "am_michael",
    speed: float = 1.0,
    lexicon_g2g: Optional[List[str]] = None,
    lexicon_g2p: Optional[List[str]] = None,
    override_author: Optional[str] = None,
    override_series: Optional[str] = None,
    output: Optional[str] = None,
    init_only=False,
    clean=False,
):
    config = ProjectConfig(
        tts_voice=voice,
        tts_speed=speed,
        lex_g2g_paths=lexicon_g2g,
        lex_g2p_paths=lexicon_g2p,
        override_author=override_author,
        override_series=override_series,
        output_path=output,
    )

    for path in paths:
        proj = AudiobookProject.open(path, config)
        proj.run_all()


def set_log_level(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logging.basicConfig(level=numeric_level)


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
    parser.add_argument(
        "--log",
        type=str,
        help="Set logging level. DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    args = parser.parse_args()

    set_log_level(args.log)

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

    logging.debug(set_args)
    logging.debug(cmd_args)
    logging.debug(cmd_metadata_overrides)
    logging.debug(cmd_only_args)

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
