# %%
from ebooklib import epub

import ebooklib

epub_path = r"../test_data/sample/sample.epub"
my_epub = epub.read_epub(epub_path)

print([f"{n}: {i.title}" for n, i in enumerate(my_epub.toc)])

my_chapter_ref = my_epub.toc[24]
print(my_chapter_ref.title, my_chapter_ref.href, my_chapter_ref.uid)

my_chapter = my_epub.get_item_with_href(my_chapter_ref.href)
# %%
toc = my_epub.toc
spine = my_epub.spine

explicit_spine = [my_epub.get_item_with_id(iid).file_name for iid, _ in spine]
indices = [explicit_spine.index(el.href) for el in toc] + [len(spine)]

chapter_groups = [explicit_spine[slice(*i)] for i in zip(indices[:-1], indices[1:])]
# for t, g in zip([t.title for t in toc], chapter_groups):
#     print(t, g)
# %%
# for g in chapter_groups:
#     for iid in g:
#         print(my_epub.get_item_with_href(iid).title)

[[my_epub.get_item_with_href(href) for href in g] for g in chapter_groups]
