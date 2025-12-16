# %%
from misaki import en, espeak
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

# %%
KOKORO_MODEL = "hexgrad/Kokoro-82M"
pipeline = KPipeline(lang_code="a", device="cuda", repo_id=KOKORO_MODEL)
# %%
text = """
Zanoba walked up to the throne and bowed deeply to his brother. Aquaharshita, Borea, and Quintellian are all Prinkes of Muragome
"""
generator = pipeline(text, voice="af_heart")
british = False
fallback = espeak.EspeakFallback(british=british)  # en-us
pipeline.g2p = en.G2P(trf=False, british=british, fallback=fallback)
# for i, (gs, ps, audio) in enumerate(generator):
#     print(i, gs, ps)
#     display(Audio(data=audio, rate=24000, autoplay=i == 0))
#     # sf.write(f"{i}.wav", audio, 24000)
# %%
import re
import wordfreq
import tqdm

with open(
    r"books\Mushoku Tensei - Volume 21 [Seven Seas][Kobo]\book.txt", encoding="utf-8"
) as f:
    full_text = f.read()

defaulted = {}
espeak_fallback = espeak.EspeakFallback(british=british)


def fallback_logger(x):
    global defaulted
    ef = espeak_fallback(x)
    defaulted[x.text.strip().lower()] = ef[0]
    return ef


g2p = en.G2P(trf=False, british=british, fallback=fallback_logger)

split_txt = re.split(r"\n+", full_text)
for text in tqdm.tqdm(split_txt):
    out = g2p(text)
    # print(out)
# %%
for k, v in defaulted.items():
    if k:
        if wordfreq.word_frequency(k, "en") < 1e-7:
            print(f'"{k}" : "{v}",')


# %%
import lxml.html
from ebooklib import epub
import cv2
from pathlib import Path
import numpy as np


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
        for entry in self.book.toc:
            title = entry.title
            item = self.book.get_item_with_href(entry.href.split("#")[0])
            chapter = Chapter(title, item)
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
        # TODO: Support multiple authors
        self.metadata["author"] = self.book.get_metadata("DC", "creator")[0][0]
        # TODO: Support publisher
        self.metadata["publisher"] = ""
        self.metadata["date"] = self.book.get_metadata("DC", "date")[0][0]
        year_match = re.match(re.compile(r"[0-9]{4}"), self.metadata["date"])
        if year_match:
            self.metadata["year"] = year_match.group(0)
        self.cover = cv2.imdecode(
            np.frombuffer(self.book.get_item_with_id("cover").content, np.uint8),
            cv2.IMREAD_COLOR,
        )


class Chapter:
    def __init__(self, title: str, item: epub.EpubItem):
        self.title = title
        self.text = self.chapter_to_text(item)

    def is_valid(self, min_length: int = 8) -> bool:
        return len(self.text) >= min_length

    @staticmethod
    def chapter_to_text(chapter: epub.EpubItem) -> str:
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

        content = chapter.get_content()
        doc = lxml.html.document_fromstring(content)
        textblocks = ["".join(tag.itertext()) for tag in doc.iter(tag="p")]
        textblocks = [t for t in textblocks if len(t)]
        textblocks = [t if not re.match(pause, t) else "<PAUSE>" for t in textblocks]
        full_text = "<BLOCK>".join(textblocks)
        for p, ss in subs:
            full_text = re.sub(p, ss, full_text)

        return full_text


tb = TextBook(
    r"books\Mushoku Tensei - Volume 21 [Seven Seas][Kobo]\Mushoku Tensei - Volume 21 [Seven Seas][Kobo].epub"
)

ch1_txt = tb.chapters[1].text
m = re.search(r"You're always cool and collected, so I feel like.+Brother", ch1_txt)
print(m.group(0))
