from typing import List, Dict, Optional, Pattern, Self, Literal
import logging
import re

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
    def from_lex(cls, path: Path):
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
                            include_ll = cls.from_lex(include_path)
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


class LexicalPatternMap(LexicalMap):
    pattern_lut: Dict[Pattern, str] | None

    def __init__(self):
        super().__init__()
        self.pattern_lut = None

    def compile(self):
        self.pattern_lut = {re.compile(k): v for k, v in self.lut.items()}

class G2PMap:
    path: Path
    map_: LexicalMap
    _note: bool = False

    def __init__(self,path:Path) -> None:
        self.path = path
        self.map_ = LexicalMap.from_lex(self.path)

    def in_map(self,k):
        return k in self.map_.lut
    
    def notate(self):
        if not self._note:
            date_str = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
            self.path.open("a",encoding="utf-8").write(f"#note {date_str}\n")

            self._note = True
    
    def add_entry(self,k,v):
        self.notate()
        self.path.open("a").write(f"{k}: {v}")

class G2GMap:
    path: Optional[Path]
    map_: LexicalPatternMap

    def __init__(self,path:Path) -> None:
        self.path = path
        self.map_ = LexicalPatternMap.from_lex(self.path)

        self.map_.compile()

class Lexicon2:
    g2g_path: Optional[Path]
    g2p_path: Optional[Path]

    g2g_map: LexicalPatternMap
    g2p_map: LexicalMap

    def __init__(self, g2g_path: Optional[Path], g2p_path: Optional[Path]) -> None:
        self.g2g_path = g2g_path
        self.g2p_path = g2p_path

        self.g2g_map = LexicalPatternMap()
        self.g2p_map = LexicalMap()
        if self.g2g_path is not None:
            self.g2g_map = LexicalPatternMap.from_lex(self.g2g_path)
        if self.g2p_path is not None:
            self.g2p_map = LexicalMap.from_lex(self.g2p_path)

    def 