# %%

from audiobooks import Book
from IPython.display import display

# from audiobook.book import Book
# from processor import AudiobookProject, Lexicon

epub_path = r"test_data\src\childrens-literature.epub"

my_book = Book(epub_path)
# my_book.meta.add_overrides(title="Boop")
display(my_book.meta)
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
