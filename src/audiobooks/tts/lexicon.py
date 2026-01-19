from typing import List, Dict, Optional, Pattern, Self, Literal, Any
import logging
import re
import attrs

import yaml
from pathlib import Path

from datetime import datetime


class SubRule:
    __slots__ = ("pattern", "replace")

    pattern: Pattern[str]
    replace: str

    def __init__(self, pattern: str, replace: str):
        self.pattern = re.compile(pattern)
        self.replace = replace

    def __call__(self, text: str) -> str:
        return self.pattern.sub(self.replace, text)


class LexicalMap:
    lut: Dict[str, str]

    MAP_PATTERN: re.Pattern = re.compile(r"^(?P<key>.+): (?P<val>.*)$")
    SPECIAL_PATTERN: re.Pattern = re.compile(r"^#(?P<cmd>\S+) (?P<args>.*)$")

    VALID_MODES = ["str", "pattern"]

    def __init__(self):
        self.lut = {}
        pass

    def keys(self):
        return self.lut.keys()

    def items(self):
        return self.lut.items()

    @staticmethod
    def process_key(k):
        return k

    @staticmethod
    def process_val(v):
        return v

    @classmethod
    def from_lut(cls, lut: Dict[str, str]):
        ll = cls()
        ll.lut = lut
        return ll

    @classmethod
    def from_file(cls, path: Path):
        if path.suffix != ".lex":
            logging.warning(f"{path} is not a *.lex")
        ll = cls()

        for l in path.open("r", encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            if l[0] == "#":
                # Special functions
                m = cls.SPECIAL_PATTERN.match(l)
                if m:
                    cmd = m.group("cmd")
                    args = m.group("args")
                    match cmd:
                        case "include":
                            # #include ./path/to/lut.lex
                            include_path = Path(args)
                            if not include_path.is_absolute():
                                include_path = path.parent / include_path
                            include_ll = cls.from_file(include_path)
                            ll.lut |= include_ll.lut
                        case _:
                            continue
                continue
            else:
                # Normal mappings
                m = cls.MAP_PATTERN.match(l)
                if m:
                    ll.lut[m.group("key")] = m.group("val")
                continue

        return ll

    def __add__(self, other: Self) -> Self:
        return self.from_lut(self.lut | other.lut)

    def __call__(self, k: str) -> Optional[str]:
        return self.lut.get(k, None)


class LexicalPatternMap(LexicalMap):
    pattern_lut: Dict[Pattern, str] | None

    def __init__(self):
        super().__init__()
        self.pattern_lut = None
        self.compile()

    @classmethod
    def from_file(cls, path: Path):
        lpm = super().from_file(path)
        lpm.compile()
        return lpm

    def compile(self):
        self.pattern_lut = {re.compile(k): v for k, v in self.lut.items()}

    def __call__(self, text: str) -> str:
        if self.pattern_lut is not None:
            for pattern, sub in self.pattern_lut.items():
                text = pattern.sub(sub, text)
        return text


class Lexicon:
    # The Lexicon can be called without either of the paths
    g2g_path: Optional[Path]
    g2p_path: Optional[Path]

    # But it will always provide at least identity transforms
    g2g: LexicalPatternMap
    g2p: LexicalMap

    def __init__(self, g2g_path: Optional[Path], g2p_path: Optional[Path]) -> None:
        self.g2g_path = g2g_path
        self.g2p_path = g2p_path

        self.g2g = (
            LexicalPatternMap.from_file(self.g2g_path)
            if self.g2g_path is not None
            else LexicalPatternMap()
        )
        self.g2p = (
            LexicalMap.from_file(self.g2p_path)
            if self.g2p_path is not None
            else LexicalMap()
        )

    def amend_g2p(self, d: Dict):
        if self.g2p_path is None:
            logging.error(
                "g2p_path is None when we're trying to write unknown words to it"
            )
            return
        sorted_d = {k: v for k, v in sorted(d.items())}
        date_str = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
        append_str = ""
        append_str += f"\n\n#unknown_words@{date_str}"
        for k, v in sorted_d.items():
            append_str += f"\n{k}: {v}"
        with self.g2p_path.open("a") as f:
            f.write(append_str)
