# %%
from ebooklib import epub

import ebooklib

from typing import cast
import lxml.html

import re

epub_path = r"../books/Ascendance of a Bookworm - Volume 02 [J-Novel Club][Premium]/Ascendance of a Bookworm - Volume 02 [J-Novel Club][Premium].epub"
# epub_path = r"..\books\Mushoku Tensei - Volume 26 [Seven Seas][Kobo]\Mushoku Tensei - Volume 26 [Seven Seas][Kobo].epub"
my_epub = epub.read_epub(epub_path)

print([f"{n}: {i.title}" for n, i in enumerate(my_epub.toc)])

my_chapter_ref = my_epub.toc[5]
print(my_chapter_ref.title, my_chapter_ref.href, my_chapter_ref.uid)

my_chapter = my_epub.get_item_with_href(my_chapter_ref.href)
# %%
toc = my_epub.toc
spine = my_epub.spine

explicit_spine = [my_epub.get_item_with_id(iid).file_name for iid, _ in spine]
indices = [explicit_spine.index(el.href.split("#")[0]) for el in toc] + [len(spine)]

chapter_groups = [explicit_spine[slice(*i)] for i in zip(indices[:-1], indices[1:])]
# for t, g in zip([t.title for t in toc], chapter_groups):
#     print(t, g)
# %%
# for g in chapter_groups:
#     for iid in g:
#         print(my_epub.get_item_with_href(iid).title)

item_groups = [[my_epub.get_item_with_href(href) for href in g] for g in chapter_groups]

chapter: epub.EpubItem = item_groups[5][0]
content = chapter.get_content()
doc = lxml.html.document_fromstring(content)
textblocks = []
textblocks += ["".join(tag.itertext()) for tag in doc.iter(tag="p")]
# %%

["".join(tag.itertext()) for tag in doc.iter(tag=["p", "h1", "h2", "h3"])]


def ttsTag(name, arg):
    return f"<{name.upper()}({arg})>"


textblocks = []
for el in doc.iter(tag=["p", "h1", "h2", "h3"]):
    txt = "".join(el.itertext())
    if el.tag == "p":
        textblocks.append(txt)
    elif m := re.match(r"h([0-9])", el.tag):
        pause = ttsTag("pause", 500)
        textblocks.append(pause)
        textblocks.append(txt)
        textblocks.append(pause)

display(textblocks)

# %%
chapter_5: epub.EpubItem = item_groups[5][0]
chapter_6: epub.EpubItem = item_groups[6][0]

h5 = lxml.html.document_fromstring(chapter_5.content)
h6 = lxml.html.document_fromstring(chapter_6.content)

h5.body.append(h6.body)

blocks = ["".join(tag.itertext()) for tag in h5.iter(tag=["h1", "p"])]

blocks

## %%
[my_epub.get_item_with_id(iid).id for iid, _ in spine]

# %%

# Get Spine and TOC in IDs
id_spine = [
    cast(epub.EpubItem, it).get_id()
    for iid, _ in my_epub.spine
    if (it := my_epub.get_item_with_id(iid)) is not None
]
id_toc = [
    cast(epub.EpubItem, it).get_id()
    for entry in my_epub.toc
    if (it := my_epub.get_item_with_href(entry.href.split("#")[0])) is not None
]
# Also get TOC titles for later
title_toc = [
    entry.title
    for entry in my_epub.toc
    if (it := my_epub.get_item_with_href(entry.href.split("#")[0])) is not None
]
# Get spine indices for each TOC item
range_indices = [id_spine.index(el) for el in id_toc]
# Split into groups bound by the indices
id_groups = [
    id_spine[slice(*i)]
    for i in zip(range_indices[:], range_indices[1:] + [len(id_spine)])
]
# Convert the id groups to item groups
item_groups: list[list[epub.EpubItem]] = [
    [
        cast(epub.EpubItem, it)
        for el in g
        if (it := my_epub.get_item_with_id(el)) is not None
    ]
    for g in id_groups
]
# Associate the titles and the item groups
list(zip(title_toc, item_groups))
