# %%

from book import Book
from processor import AudiobookProject, Lexicon

# my_book = Book(r"test_data\test\test.epub")
print("Tester")
my_lexicon = Lexicon(
    g2g_paths=[
        r"G:\Projects\audiobooks\lexicons\g2g\symbols.yaml",
        r"G:\Projects\audiobooks\lexicons\g2g\exclamations.yaml",
        r"G:\Projects\audiobooks\lexicons\g2g\AoaB.yaml",
    ],
    g2p_paths=[
        r"G:\Projects\audiobooks\lexicons\g2p\AoaB.yaml",
    ],
)
my_proj = AudiobookProject(r"G:\Projects\audiobooks\test_data\test", my_lexicon)
print(my_proj.lexicon)
