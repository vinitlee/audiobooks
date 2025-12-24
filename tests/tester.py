# %%

from book import Book
from processor import AudiobookProject, Lexicon

epub_path = r"../books/Ascendance of a Bookworm - Volume 02 [J-Novel Club][Premium]/Ascendance of a Bookworm - Volume 02 [J-Novel Club][Premium].epub"

my_book = Book(epub_path)
# print("Tester")
# my_lexicon = Lexicon(
#     g2g_paths=[
#         r"..\lexicons\g2g\symbols.yaml",
#         r"..\lexicons\g2g\exclamations.yaml",
#         r"..\lexicons\g2g\AoaB.yaml",
#     ],
#     g2p_paths=[
#         r"..\lexicons\g2p\AoaB.yaml",
#     ],
# )
# my_proj = AudiobookProject(r"./test_data/sample")
# my_proj.make_splits()
