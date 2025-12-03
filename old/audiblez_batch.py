# %%
import subprocess
from pathlib import Path
import os
import re
from ebooklib import epub
import json

BASE_PATH = Path(__file__).parent
os.chdir(BASE_PATH)
BOOKS_DIR = Path("books")
AUDIBLEZ_PROJECT = Path(r"G:\Projects\audiblez-app")

VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santabf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santaff_siwishf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psiif_sara",
    "im_nicolajf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumopf_dora",
    "pm_alex",
    "pm_santazf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]


def get_command(
    input_file: Path, output_dir: Path, voice: str = "bm_lewis", speed: float = 1
):
    if voice not in VOICES:
        raise ValueError(f"{voice} not available.")

    return [
        "uv",
        "--project",
        str(AUDIBLEZ_PROJECT),
        "run",
        "audiblez",
        "-c",
        "-v",
        voice,
        "-s",
        f"{speed}",
        "-o",
        f"{str(output_dir.resolve())}",
        f"{str(input_file.resolve())}",
    ]

    # return ["uv", "--project", str(AUDIBLEZ_PROJECT), "run", "audiblez", "--help"]


def get_wav_chapters(project_dir: Path):
    chapters = {}
    chapter_paths = list(project_dir.glob("*chapter*.wav"))
    for p in chapter_paths:
        m = re.match(re.compile(r"^.+?chapter_([0-9]+)_"), str(p))
        if m:
            chapters[int(m.group(1))] = p

    return chapters


def make_better_m4b(project_dir: Path):
    wavs = get_wav_chapters(project_dir)
    wavs = [wavs[k] for k in sorted(wavs.keys())]
    print(wavs)

    book = epub.read_epub("book.epub")
    chapters = []
    for item in book.get_items():
        if item.get_type() == epub.EpubHtml:
            print("")


def convert_m4b(project_dir: Path):

    m4b = list(project_dir.glob("*.m4b"))
    if len(m4b) > 1:
        print(f"Warning: m4b has multiple matches: {m4b}")
    m4b = m4b[0].resolve()

    command = [
        "ffmpeg",
        "-i",
        str(m4b),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-aac_coder",
        "twoloop",
        "-ar",
        "24000",
        "-threads",
        "0",
        "-read_ahead_limit",
        "0",
        "-thread_queue_size",
        "2048",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        str(m4b.with_suffix(".x.m4b")),
    ]

    print(command)

    try:
        subprocess.run(command)
    except:
        print(f"There was a problem running conversion on {project_dir}")
        return False
    return True


def main():
    for epub in BOOKS_DIR.glob("*.epub"):
        project_dir = BOOKS_DIR / epub.stem
        if project_dir.exists():
            print(f"Skipping {epub}, directory exists.")
            continue
        project_dir.mkdir(parents=True, exist_ok=False)
        epub = epub.rename(project_dir / epub.name)

    for project_dir in [p for p in BOOKS_DIR.glob("*") if p.is_dir()]:
        epub = list(project_dir.glob("*.epub"))[0]

        tts_flag = project_dir / "flag_tts"
        if not tts_flag.exists():
            command = get_command(epub, project_dir, speed=1.3)
            subprocess.run(command)
            json.dump(
                {"epub": str(epub), "m4b": str(epub.with_suffix(".m4b"))},
                tts_flag.open("w"),
            )

        conv_check = project_dir / "flag_conv"
        if not conv_check.exists():
            success = False
            # make_better_m4b(project_dir)
            success = convert_m4b(project_dir)
            if success:
                conv_check.write_text("")


if __name__ == "__main__":
    main()

# """
# ffmpeg -i audiblez_output.m4b -c:a aac -b:a 64k -map_metadata 0 -map_chapters 0 final.m4b

# ffmpeg -i ".\Mushoku Tensei - Volume 04 [Seven Seas][Kobo].m4b" -map 0:a:0 -c:a aac -b:a 64k -map_metadata 0 -map_chapters 0 final.m4b

# ffmpeg -i ".\Mushoku Tensei - Volume 04 [Seven Seas][Kobo].m4b" -map 0:a:0 -map 0:v -c:v copy -disposition:v:0 attached_pic -sn -dn -c:a aac -b:a 64k -aac_coder twoloop -ar 24000 -threads 0 -map_metadata 0 -map_chapters 0 ".\Mushoku Tensei - Vol 04.m4b"

# ffmpeg -i ".\Mushoku Tensei - Volume 04 [Seven Seas][Kobo].m4b" -map 0:a:0 -vn -sn -dn -c:a aac -b:a 64k -aac_coder twoloop -ar 24000 -threads 0 -read_ahead_limit 0 -thread_queue_size 2048 -map_metadata 0 -map_chapters 0 ".\Mushoku Tensei - Vol 04.m4b"
# """
