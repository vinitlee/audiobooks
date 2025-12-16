# %%
from misaki import en, espeak
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import re
import wordfreq
import tqdm
import lxml.html
from ebooklib import epub
import cv2
from pathlib import Path
import numpy as np
import json
import yaml
import glob


def load_config(config_path, valid_args):
    config = yaml.safe_load(Path(config_path).open())
    valid_config = {k: v for k, v in config.items() if (k in valid_args)}

    return valid_config


# %%
config_path = Path(r"configs\rezeroex_config.yaml")
config = load_config(config_path, ["files", "lexicon"])

lexicon_path = Path(config["lexicon"])
lexicon = json.load(lexicon_path.open(encoding="utf-8"))

expanded_paths = set()
for p in config["files"]:
    expanded_paths = expanded_paths.union(glob.glob(p))
book_paths = [Path(p) / "book.txt" for p in expanded_paths]
# %%
if not lexicon:
    full_text = "\n\n".join([p.open(encoding="utf-8").read() for p in book_paths])

    defaulted = {}
    espeak_fallback = espeak.EspeakFallback(british=False)

    def fallback_logger(x):
        global defaulted
        ef = espeak_fallback(x)
        defaulted[x.text.strip().lower()] = ef[0]
        return ef

    g2p = en.G2P(trf=False, british=False, fallback=fallback_logger)

    split_txt = re.split(r"\n+", full_text)
    for text in tqdm.tqdm(split_txt):
        out = g2p(text)
    stripped_defaulted = {}
    for k in sorted(list(defaulted.keys())):
        if k:
            if wordfreq.word_frequency(k, "en") < 1e-7:
                # print(f'"{k}" : "{defaulted[k]}",')
                stripped_defaulted[k] = defaulted[k]

    stripped_defaulted.update(lexicon)
    json.dump(
        stripped_defaulted,
        lexicon_path.open(mode="w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )

# %%
KOKORO_MODEL = "hexgrad/Kokoro-82M"
pipeline = KPipeline(lang_code="a", device="cuda", repo_id=KOKORO_MODEL)
# %%


def pronounce(grapheme: str | None = None, phoneme: str | None = None, verbose=True):
    pronounce_str: str = ""
    if phoneme is None:
        if not grapheme:
            grapheme = "x"
        pronounce_str = grapheme or ""
    else:
        pronounce_str = f"[{grapheme}](/{phoneme}/)"
    generator = pipeline(pronounce_str, voice="af_heart")
    for i, (gs, ps, audio) in enumerate(generator):
        if verbose:
            print(f'pronounce("{gs}","{ps}")')
            print(f'"{gs}":"{ps}",')
        display(Audio(data=audio, rate=24000, autoplay=False))


def pronounce_from_lexicon(w: str, verbose=True):
    pronounce(w, lexicon[w], verbose=verbose)


# %%
def pronounce_section(n, step=10):
    lexicon = json.load(lexicon_path.open(encoding="utf-8"))
    i = n * step
    for w in list(lexicon.keys())[i : i + step]:
        pronounce(w, lexicon[w], verbose=True)
