from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from typing import Any, Callable, Iterable, Optional
    from pathlib import Path

import yaml


def main(
    files: set[Path | str],
    voice: str = "am_michael",
    speed: float = 1,
    lexicon: Path | str | None = None,
    init_only=False,
    output: Path | str | None = None,
    overwrite_project=False,
    overwrite_tts=False,
    overwrite_master=False,
    overwrite_m4b=False,
    overwrite_output=False,
    clean=False,
):

    # print(files)
    print(output)

    for p in files:
        my_proj = TTSProject(p, lexicon=lexicon, voice=voice, speed=speed)
        my_proj.save_book_text()
        # TODO: There are probably too many overwrite flags, figure out if you want overwrite_project or just sections
        if (not my_proj.completed) or overwrite_project:
            if not init_only:
                my_proj.generate_sound_files(overwrite=overwrite_tts)
                my_proj.create_ffmetadata()
                my_proj.build_master_audio(overwrite=overwrite_master)
                my_proj.build_m4b(overwrite=overwrite_m4b)
        else:
            print(f"Not processing {p}, already completed.")
        if output and my_proj.completed:
            my_proj.copy_to_library(output, overwrite=overwrite_output, epub=True)
            if clean:
                print("Clean not implemented.")


def load_config(config_path, valid_args):
    config = yaml.safe_load(Path(config_path).open())
    valid_config = {k: v for k, v in config.items() if (k in valid_args)}

    return valid_config


def parse_arguments() -> dict:
    set_args = {}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        metavar="PATH",
        nargs="*",
        help="Path(s) to *.epub or dir to be processed; accepts * as a wildcard",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        help="Kokoro voice",
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
        help="TTS Speed",
    )
    parser.add_argument(
        "--lexicon",
        "-lx",
        type=str,
        help="Path to external lexicon file",
    )
    parser.add_argument(
        "--init_only",
        "-ino",
        action="store_true",
        help="Just set up project, don't generate",
    )
    parser.add_argument(
        "--overwrite_project",
        "-owp",
        action="store_true",
        help="Allow overwriting project",
    )
    parser.add_argument(
        "--overwrite_tts",
        "-owt",
        action="store_true",
        help="Allow overwriting TTS",
    )
    parser.add_argument(
        "--overwrite_master",
        "-owm",
        action="store_true",
        help="Allow overwriting concatenated audio master",
    )
    parser.add_argument(
        "--overwrite_m4b",
        "-owb",
        action="store_true",
        help="Allow overwriting M4B",
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
        "--config",
        "-c",
        metavar="PATH",
        help="Specify YAML config. YAML will be overridden by any cmd arguments.",
    )
    args = parser.parse_args()

    set_args = {}
    if args.config:
        set_args.update(load_config(args.config, vars(args).keys()))
    set_args.update(
        {
            k: v
            for k, v in vars(args).items()
            if (bool(v) and (v is not None) and (v is not False))
        }
    )
    del set_args["config"]

    # print(set_args)
    expanded_paths = set()
    for p in set_args["files"]:
        expanded_paths = expanded_paths.union(glob.glob(p))
    set_args["files"] = expanded_paths
    # print(set_args)

    return set_args


my_proj = None
if __name__ == "__main__":
    arguments = parse_arguments()

    main(**arguments)
