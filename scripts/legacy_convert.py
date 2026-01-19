# %%
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from typing import Any, Callable

from line_profiler import profile
from pathlib import Path
import glob
from pathvalidate import sanitize_filepath
import json
import re
import subprocess
import datetime
import shutil
import itertools

import argparse
import yaml

import numpy as np
import cv2

import soundfile as sf

from ebooklib import epub
import ebooklib
import lxml.html
import torch

import warnings
from tqdm import tqdm


# %%
class KPipelineLazy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._obj = None

    def load(self):
        if self._obj is None:
            from kokoro import KPipeline

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                self._obj = KPipeline(*self.args, **self.kwargs)
        return self._obj

    def __getattr__(self, name: str) -> profile:
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)


PIPELINE = KPipelineLazy(lang_code="a", device="cuda", repo_id="hexgrad/Kokoro-82M")
# if torch.cuda.is_available():
#     PIPELINE = KPipelineLazy(lang_code="a", device="cuda", repo_id="hexgrad/Kokoro-82M")
# else:
#     PIPELINE = KPipelineLazy(lang_code="a", device="cpu", repo_id="hexgrad/Kokoro-82M")


class TTSProject:
    epub_path: Path
    lexicon: dict | None = None
    lexicon_path: Path | None = None

    def __init__(
        self,
        starting_path: Path | str,
        pipeline: KPipeline | None = None,
        lexicon: Path | str | None = None,
        voice: str = "bm_lewis",
        speed: float = 1,
    ):
        self.project_dir_path: Path

        if lexicon is not None:
            self.lexicon_path = Path(lexicon)

        starting_path = Path(starting_path)
        if not starting_path.exists():
            raise FileExistsError(f"{starting_path.resolve()} does not exist")

        if starting_path.is_dir():
            self.load_project_dir(starting_path)
        elif starting_path.suffix == ".epub":
            self.init_project_dir(starting_path)

        if pipeline is None:
            global PIPELINE
            self.pipeline = PIPELINE
        else:
            self.pipeline = pipeline

        self.data["voice"] = voice

        self.data["speed"] = speed

        self.book: TextBook = TextBook(self.epub_path)

        m4b_path = self.project_dir_path / sanitize_filepath(
            str(self.book.title).replace(".", "") + ".m4b"
        )
        self.data["m4b_path"] = str(m4b_path.resolve())

        self.write_data_file()

    # def setup_paths(self):

    @property
    def completed(self) -> bool:
        return self.data.get("completed", False)

    def save_book_text(self, output="book.txt"):
        text = "\n\n".join([c.text for c in self.book.chapters])
        (self.project_dir_path / output).write_text(text, encoding="utf-8")

    def lexicon_g2p(self, stock_g2p):
        def custom_g2p(text: str):
            gs, tokens = stock_g2p(text)

            if tokens is not None:
                for t in tokens:
                    key = t.text.lower()
                    if key in self.lexicon:
                        t.phonemes = self.lexicon[key]

            return gs, tokens

        return custom_g2p

    def generate_sound_files(self, overwrite=False):
        # TODO: Make skipping more efficient by reading data rather than wiping it
        self.data["chapters"] = []
        self.data["generated"] = False

        # Swap out G2P
        stock_g2p = self.pipeline.g2p
        self.pipeline.g2p = self.lexicon_g2p(stock_g2p)

        for i, chapter in enumerate(self.book.chapters):

            output_path = self.project_dir_path / f"split-{i}.wav"

            if (not output_path.exists()) or overwrite:
                sr = 24000
                split_pattern = r"\n+"
                generator = self.pipeline(
                    chapter.text,
                    voice=self.data["voice"],
                    speed=self.data["speed"],
                    split_pattern=split_pattern,
                )
                audio_clips = []

                segments = [
                    s for s in re.split(split_pattern, chapter.text) if s.strip()
                ]
                segment_word_counts = [len(seg.split()) for seg in segments]
                total_words = sum(segment_word_counts)
                word_pbar = tqdm(total=total_words, desc=chapter.title, unit="words")
                word_count_iter = iter(segment_word_counts)
                for gph, phn, snd in generator:
                    if type(snd) in [torch.FloatTensor, torch.Tensor]:
                        audio_clips.append(snd.detach().cpu().numpy())
                    word_pbar.update(next(word_count_iter, 0))
                word_pbar.close()
                full_audio = np.concat(audio_clips)  # TODO: join with adustable gap
                scale = 1.175
                max_scale = 1 / max(abs(full_audio.min()), abs(full_audio.max()))
                # if max_scale < scale:
                #     print("Choosing lower scale: ", max_scale)
                scale = min(scale, max_scale)
                # print(scale, max_scale)
                full_audio *= scale
                sf.write(output_path, full_audio, sr)
            else:
                print(f"Chapter[{i}] output exists, skipping.")
                full_audio, sr = sf.read(str(output_path))

            self.data["chapters"].append(
                {
                    "number": i + 1,
                    "title": chapter.title,
                    "length": int(
                        1000 * len(full_audio) / sr,
                    ),
                    "path": str(output_path.resolve()),
                }
            )

        # Restore original G2P
        self.pipeline.g2p = stock_g2p

        self.data["generated"] = True
        self.write_data_file()

    def build_master_audio(self, overwrite=True):
        output_path = self.project_dir_path / "master.m4a"
        fileslist_path = self.project_dir_path / "files.txt"
        if (not any([output_path.exists(), fileslist_path.exists()])) or overwrite:
            output_path.unlink(missing_ok=True)

            fileslist = []
            for chapter in self.data.get("chapters", []):
                path = Path(chapter.get("path", "")).as_posix()
                if path:
                    fileslist.append(f"file '{path}'")
                else:
                    print(f"No path for {chapter.get('number',-1)}")
            fileslist_path.write_text("\n".join(fileslist))

            cmd: list[str] = ["ffmpeg"]
            cmd += ["-f", "concat"]
            cmd += ["-safe", "0"]
            cmd += ["-i", str(fileslist_path.resolve())]
            cmd += ["-c:a", "aac"]  # AAC encoder
            cmd += ["-aac_coder", "fast"]  # AAC fast encoder
            cmd += ["-b:a", "48k"]  # bitrate
            cmd += ["-ac", "1"]  # channels
            cmd += ["-ar", "24000"]  # sample rate
            cmd += ["-y"]  # overwrite without prompt
            cmd += [str(output_path.resolve())]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                process.stdout.close()
            process.wait()

            self.data["files_list_path"] = str(fileslist_path.resolve())
            self.data["audio_path"] = str(output_path.resolve())
            self.write_data_file()
        else:
            print("Master audio exists, skipping.")

    def save_cover_image(self):
        cover_path = self.project_dir_path / "cover.jpg"
        cv2.imwrite(str(cover_path), self.book.cover)
        self.data["cover_path"] = str(cover_path.resolve())
        self.write_data_file()

    def build_m4b(self, overwrite=False):

        if (not self.data.get("completed", False)) or overwrite:
            self.save_cover_image()
            m4b_path = self.data.get("m4b_path", "")
            audio_path = self.data.get("audio_path", "")
            ffmetadata_path = self.data.get("ffmetadata_path", "")
            cover_path = self.data.get("cover_path", "")

            if not all([audio_path, cover_path, ffmetadata_path, m4b_path]):
                print("audio_path", audio_path)
                print("cover_path", cover_path)
                print("ffmetadata_path", ffmetadata_path)
                print("m4b_path", m4b_path)
                raise Exception("Build M4B: Some path is empty or missing")

            cmd: list[str] = ["ffmpeg"]
            cmd += ["-i", audio_path]
            cmd += ["-i", cover_path]
            cmd += ["-i", ffmetadata_path]
            cmd += ["-map", "0:a"]  # audio_path -> audio
            cmd += ["-map", "1:v"]  # "video" is actually the cover
            cmd += ["-c:a", "copy"]  # just copy audio
            cmd += ["-c:v", "copy"]  # just copy video
            cmd += ["-map_metadata", "2"]  # pull metadata from ffmetadata
            cmd += ["-map_chapters", "2"]  # pull chapters from ffmetadata
            cmd += ["-metadata:s:v:0", "title=Cover"]
            cmd += ["-metadata:s:v:0", "comment=Cover (front)"]
            cmd += ["-disposition:v:0", "attached_pic"]
            cmd += ["-movflags", "+faststart"]  # stream-friendly
            cmd += [m4b_path]

            Path(m4b_path).unlink(missing_ok=True)
            subprocess.run(cmd, check=True)

            self.data["completed"] = True
            self.write_data_file()

    def copy_to_library(self, parent_dir: Path | str, overwrite=False, epub=False):
        parent_dir = Path(parent_dir).expanduser().resolve()
        m4b_src = Path(self.data.get("m4b_path", ""))
        if not str(m4b_src):
            print("Source m4b not found, aborting.")
            pass

        book_dir = parent_dir / m4b_src.stem
        print("book_dir is ", book_dir)
        book_dir.mkdir(parents=True, exist_ok=True)
        m4b_filename = m4b_src.name
        m4b_dest = book_dir / m4b_filename
        if not m4b_dest.exists() or overwrite:
            shutil.copy(m4b_src, m4b_dest)
        else:
            print(f"{m4b_dest} already exists. Skipping copy.")

        if epub:
            epub_dest = book_dir / self.epub_path.name
            if not epub_dest.exists() or overwrite:
                shutil.copy(self.epub_path, epub_dest)
            else:
                print(f"{epub_dest} already exists. Skipping copy.")

    def create_ffmetadata(self):
        header_data = {
            "title": self.book.title,
            "artist": self.book.author,
            # "album": self.book.title,
            # "album_artist": {self.book.author},
            "genre": "Audiobook",
            "date": self.book.metadata.get("year", ""),
            "comment": f"Generated by vinitlee on {str(datetime.date.today())}",
        }
        header = f";FFMETADATA1\n"
        for k, v in header_data.items():
            if v:
                header += f"{k}={v}\n"

        def chapter_split_text(start, end, title):
            split = "\n[CHAPTER]\nTIMEBASE=1/1000\n"
            split += f"START={start}\n"
            split += f"END={end}\n"
            split += f"title={title}\n"
            return split

        chapter_splits = []
        current_time = 0
        for ch in self.data.get("chapters", []):
            start = current_time
            end = current_time + ch.get("length")
            title = ch.get("title", f"Chapter {ch.get('number','X')}")
            chapter_splits.append(chapter_split_text(start, end, title))

            current_time = end

        ffmetadata = "".join([header] + chapter_splits)

        ffmetadata_path = self.project_dir_path / "ffmetadata"
        ffmetadata_path.write_text(ffmetadata, encoding="UTF-8")
        self.data["ffmetadata_path"] = str(ffmetadata_path.resolve())
        self.write_data_file()

    def load_project_dir(self, project_dir: Path):
        epub_files = list(project_dir.glob("*.epub"))
        if len(epub_files) > 1:
            warnings.warn(f"Multiple EPUBs found, defaulting to {epub_files[0].name}")
        self.project_dir_path = project_dir
        self.epub_path = epub_files[0]

        self.lexicon_path = self.lexicon_path or (project_dir / "lexicon.json")
        self.lexicon = {}
        if self.lexicon_path.exists():
            self.lexicon = json.load(self.lexicon_path.open("r", encoding="utf-8"))
        else:
            self.write_lexicon_file()

        self.data_path = project_dir / "data.json"
        self.data = {}
        if self.data_path.exists():
            self.data = json.load(self.data_path.open("r", encoding="utf-8"))
        else:
            self.write_data_file()

    def add_lexicon_entry(self, key, value):
        self.lexicon[key] = value
        self.write_lexicon_file()

    def remove_lexicon_entry(self, key):
        del self.lexicon[key]
        self.write_lexicon_file()

    def write_data_file(self):
        json.dump(self.data, self.data_path.open("w"), indent=4)

    def write_lexicon_file(self):
        if self.lexicon_path is not None:
            json.dump(self.lexicon, self.lexicon_path.open("w"), indent=4)

    def init_project_dir(self, epub_file: Path):
        new_dir = Path(str(epub_file.with_suffix("")) + "/")
        if not new_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=False)
            epub_file.rename(new_dir / epub_file.name)
        else:
            print(f"{new_dir} already exists, loading that project instead.")

        self.load_project_dir(new_dir)


class TextBook:
    def __init__(
        self,
        epub_path: Path,
    ):
        self.book = epub.read_epub(epub_path)
        self.metadata: dict[str, str | int | float] = {}
        self.unpack_book_metadata()
        self.chapters: list[Chapter] = []
        self.unpack_chapters()

    @property
    def title(self):
        return self.metadata.get("title", "-title-")

    @property
    def author(self):
        return self.metadata.get("author", "-author-")

    def unpack_chapters(self):
        # Spine with href-style ids
        flat_toc = []
        for el in self.book.toc:
            if isinstance(el, tuple):
                print("tuple in toc detected: first element is ", el[0])
                flat_toc.extend(el[1])
            else:
                flat_toc.append(el)

        explicit_spine = [
            self.book.get_item_with_id(iid).file_name for iid, _ in self.book.spine
        ]
        range_indices = [explicit_spine.index(el.href.split("#")[0]) for el in flat_toc]

        # Groups of chapters that correspond with the TOC entries
        href_groups = [
            explicit_spine[slice(*i)]
            for i in zip(range_indices[:], range_indices[1:] + [len(explicit_spine)])
        ]

        item_groups = [
            [self.book.get_item_with_href(href) for href in g] for g in href_groups
        ]

        for i, entry in enumerate(flat_toc):
            title = entry.title
            # item = self.book.get_item_with_href(entry.href.split("#")[0])
            items = item_groups[i]
            chapter = Chapter(title, items)
            if chapter.is_valid():
                self.chapters.append(chapter)
        # Hack to get the title at the beginning
        self.chapters[0].text = (
            f"{self.title} by {self.author}\n" + self.chapters[0].text
        )

    def unpack_book_metadata(self):
        self.metadata = {
            "title": "",
            "author": "",
            "publisher": "",
            "date": "",
            "year": "",
        }
        self.metadata["title"] = self.book.title
        try:
            # TODO: Support multiple authors
            self.metadata["author"] = self.book.get_metadata("DC", "creator")[0][0]
        except:
            pass
        # TODO: Support publisher
        self.metadata["publisher"] = ""
        try:
            self.metadata["date"] = self.book.get_metadata("DC", "date")[0][0]
            year_match = re.match(re.compile(r"[0-9]{4}"), self.metadata["date"])
            if year_match:
                self.metadata["year"] = year_match.group(0)
        except:
            pass

        cover_items = list(self.book.get_items_of_type(ebooklib.ITEM_COVER))
        self.cover = np.zeros((600, 800, 3), dtype=np.uint8)
        try:
            self.cover = cv2.imdecode(
                np.frombuffer(cover_items[0].content, np.uint8),
                cv2.IMREAD_COLOR,
            )
        except:
            print("Could not retrieve cover.")
            print(cover_items)


class Chapter:
    def __init__(self, title: str, item: epub.EpubItem):
        self.title = title
        self.text = self.chapter_to_text(item)

    def is_valid(self, min_length: int = 8) -> bool:
        return len(self.text) >= min_length

    @staticmethod
    def chapter_to_text(chapters: list[epub.EpubItem]) -> str:
        pause = re.compile(r"^[ \xa0(***)]+$")
        subs = [
            (re.compile(r"(<BLOCK><PAUSE>)+<BLOCK>"), "<BLOCK>……………<BLOCK>"),
            (re.compile(r"<BLOCK>"), "\n"),
            (re.compile(r"(<PAUSE>)+"), "\n"),
            (re.compile(r"\:"), ": "),
            (re.compile(r"[\“\”\"]"), '"'),
            (re.compile(r"[\‘\’\']"), "'"),
            (re.compile(r"\xa0+"), " "),
            (re.compile(r"[*~@<>_+=]"), ""),
            (re.compile(r"hk"), "hek"),
            (re.compile(r"\n…"), "…"),
            (re.compile(r"\n+"), "\n"),
        ]

        textblocks = []

        for chapter in chapters:
            content = chapter.get_content()
            doc = lxml.html.document_fromstring(content)
            textblocks += ["".join(tag.itertext()) for tag in doc.iter(tag="p")]
        textblocks = [t for t in textblocks if len(t)]
        textblocks = [t if not re.match(pause, t) else "<PAUSE>" for t in textblocks]
        full_text = "<BLOCK>".join(textblocks)
        for p, ss in subs:
            full_text = re.sub(p, ss, full_text)

        return full_text


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
    # print(output)

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
    if "config" in set_args:
        del set_args["config"]

    # print(set_args)
    expanded_paths = set()
    for p in set_args["files"]:
        g = glob.glob(str(Path(p).expanduser().resolve()))
        expanded_paths = expanded_paths.union(g)
    set_args["files"] = sorted(expanded_paths)
    # print(set_args)

    return set_args


my_proj = None
if __name__ == "__main__":
    arguments = parse_arguments()

    # print(arguments)

    main(**arguments)
# %%
