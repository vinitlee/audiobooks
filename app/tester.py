# %%

from book import Book
from processor import AudiobookProject, Lexicon

# my_book = Book(r"test_data\test\test.epub")
print("Tester")
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
my_proj = AudiobookProject(r"./test_data/sample")
my_proj.make_splits()
