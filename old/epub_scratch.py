# %%
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import numpy as np
import os
import ebooklib
from ebooklib import epub
import numpy as np
from html.parser import HTMLParser
import lxml.html
from lxml import etree
import re
import cv2
import json
from pathlib import Path

if not torch.cuda.is_available():
    raise Exception("CUDA not available")

# %%

KOKORO_VOICES = [
    "af_bella",  # feels kinda tiktok-ish
    "af_nicole",  # way too asmr
    "af_sarah",  # very clear but a little flat
    "af_sky",  # ok
    "bf_emma",  # ok but a tad flat
    "bf_isabella",  # ok but a little too scottish
    "am_adam",  # this is the tiktok voice
    "am_michael",  # good
    "bm_george",  # definitely fine, but feels like it loses out against lewis
    "bm_lewis",  # good
]
KOKORO_MODEL = "hexgrad/Kokoro-82M"


def init_kokoro(lexicon={}):
    pipeline = KPipeline(lang_code="a", device="cuda", repo_id=KOKORO_MODEL)
    init_g2p(pipeline, lexicon=lexicon)
    return pipeline


def init_g2p(pipeline, lexicon={}):
    stock_g2p = pipeline.g2p

    def custom_g2p(text: str):
        gs, tokens = stock_g2p(text)

        if tokens is not None:
            # Lexicon
            for t in tokens:
                key = t.text.lower()
                if key in lexicon:
                    t.phonemes = lexicon[key]

        return gs, tokens

    pipeline.g2p = custom_g2p


def tts(
    pipeline, text, output, voice="bm_lewis", speed=1.0, split_pattern=r"\n+", gap=0
):
    """
    Takes a block of text, processes each split separately, joins them, and saves the output wav
    """
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
    audio_clips = []
    for sec in generator:
        gph, phn, snd = sec
        # print(phn)
        if type(snd) in [torch.FloatTensor, torch.Tensor]:
            audio_clips.append(snd.detach().cpu().numpy())
    full_audio = np.concat(audio_clips)
    # TODO: join with gap
    sf.write(output, full_audio, 24000)


class TTS_Book:
    @staticmethod
    def chapter_to_text(chapter: epub.EpubItem):
        pause = re.compile(r"^[ \xa0(***)]+$")
        subs = [
            (re.compile(r"(<BLOCK><PAUSE>)+<BLOCK>"), "<BLOCK>……………<BLOCK>"),
            (re.compile(r"<BLOCK>"), "\n"),
            (re.compile(r"(<PAUSE>)+"), "\n"),
            (re.compile(r"\:"), ": "),
            (re.compile(r"[\“\”\"]"), '"'),
            (re.compile(r"[\‘\’\']"), "'"),
            (re.compile(r"[\xa0*~@<>_+=]"), ""),
            (re.compile(r"hk"), "hek"),
            (re.compile(r"\n…"), "…"),
            (re.compile(r"\n+"), "\n"),
        ]

        content = chapter.get_content()
        doc = lxml.html.document_fromstring(content)
        title = doc.xpath("//title")
        # print(content)
        textblocks = ["".join(tag.itertext()) for tag in doc.iter(tag="p")]
        textblocks = [t for t in textblocks if len(t)]
        textblocks = [t if not re.match(pause, t) else "<PAUSE>" for t in textblocks]
        full_text = "<BLOCK>".join(textblocks)
        for p, ss in subs:
            full_text = re.sub(p, ss, full_text)

        return full_text

    def __init__(self, epub_path):
        self.book = epub.read_epub(epub_path)

        self.title = self.book.title
        self.author = self.book.get_metadata("DC", "creator")[0][0]

        self.cover = cv2.imdecode(
            np.frombuffer(self.book.get_item_with_id("cover").content, np.uint8),
            cv2.IMREAD_COLOR,
        )

        # self.spine_items = [self.book.get_item_with_id(id) for id, _ in self.book.spine]
        # self.toc_items = [
        #     self.book.get_item_with_href(entry.href.split("#")[0])
        #     for entry in self.book.toc
        # ]
        self.chapter_titles = []
        self.chapter_texts = []

        for entry in self.book.toc:
            title = entry.title
            item = self.book.get_item_with_href(entry.href.split("#")[0])
            text = self.chapter_to_text(item)
            if len(text) > 8:  # small number
                self.chapter_titles.append(title)
                self.chapter_texts.append(text)

    @property
    def chapters(self):
        val = [
            (self.chapter_titles[i], self.chapter_texts[i])
            for i in range(len(self.chapter_titles))
        ]
        print(val)
        return val


# %%
pipeline = init_kokoro(json.load(Path(r"lexicons/mushoku_tensei.json").open()))

# %%
test_book = TTS_Book(Path("test_data/test.epub"))
sample_title, sample_text = test_book.chapters[4]
sample_text = "\n".join(sample_text.split("\n")[:16])
print(sample_text)

tts(pipeline, sample_text, r"test_data/mushoku1.wav")
