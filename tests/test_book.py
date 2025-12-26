from audiobooks import Book


def test_book():
    my_book = Book(r"test_data/src/childrens-literature.epub")
    assert my_book.meta.title == "Children's Literature"
