from audiobooks.tts import LexicalMap
from pathlib import Path

p = Path(r"G:\Projects\audiobooks\lexicons\lex\AoaB.lex")

ll = LexicalMap.from_file(p)

for k, v in ll.lut.items():
    print(k, v)
