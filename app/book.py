from __future__ import annotations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Dict

from pathlib import Path
import glob
import re
import warnings

import json
import yaml

import numpy as np
import cv2

from ebooklib import epub
import ebooklib
import lxml.html


class Book:
    def __init__(
        self,
        epub_path: Path | str,
    ):
        self.epub = epub.read_epub(epub_path)

        self.meta = BookMetadata(self.epub)
        self.chapters: list[Chapter]
        self.init_chapters()

    def init_chapters(self):
        # Get Spine and TOC in IDs
        id_spine = [
            cast(epub.EpubItem, it).get_id()
            for iid, _ in self.epub.spine
            if (it := self.epub.get_item_with_id(iid)) is not None
        ]
        id_toc = [
            cast(epub.EpubItem, it).get_id()
            for entry in self.epub.toc
            if (it := self.epub.get_item_with_href(entry.href.split("#")[0]))
            is not None
        ]
        # Also get TOC titles for later
        title_toc = [
            entry.title
            for entry in self.epub.toc
            if (it := self.epub.get_item_with_href(entry.href.split("#")[0]))
            is not None
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
                if (it := self.epub.get_item_with_id(el)) is not None
            ]
            for g in id_groups
        ]
        # Associate the titles and the item groups
        # and create Chapter
        self.chapters = [
            ch
            for title, item_group in zip(title_toc, item_groups)
            if (ch := Chapter(title, item_group)).is_valid()
        ]

        self.chapters[0].insert(0, f"{self.meta.title} by {self.meta.author}", "h1")

    @property
    def fulltext(self, delimiter: str = "", chapter_delimiter: str = ""):
        return delimiter.join([c.fulltext(chapter_delimiter) for c in self.chapters])

    def __repr__(self) -> str:
        return f"{self.meta.title}: {len(self.chapters)} chapters"


class BookMetadata:
    _delim: str = ", "
    _overrides: Dict[str, Any] = {}

    def __init__(self, epub_obj: epub.EpubBook):
        self.get = epub_obj.get_metadata

        # Esential attributes
        self.title = epub_obj.title

        # Cover
        cover_items = list(epub_obj.get_items_of_type(ebooklib.ITEM_COVER))
        cover_items += [
            epub_obj.get_item_with_id(i.id) for i in epub_obj.items if "cover" in i.id
        ]
        if len(cover_items):
            self.cover = cv2.imdecode(
                np.frombuffer(cover_items[0].content, np.uint8),
                cv2.IMREAD_COLOR,
            )
        else:
            warnings.warn("Could not get cover image.")

        # Inferred tags
        self._tags = set()
        for namespace, items in epub_obj.metadata.items():
            for name in items:
                self._tags.add(name)

    # One-off cases here
    @property
    def year(self):
        # Try to pull from rights
        m = re.search(r"[0-9]{4}", str(self.rights))
        if m:
            return m.group(0)
        return ""

    @property
    def author(self):
        return self.creator

    def __getattr__(self, name):
        # Handles general cases
        if name in self._overrides:
            return self._overrides.get(name)
        entry = self.get("DC", name)
        if entry and len(entry):
            vals, ids = zip(*entry)
            return self._delim.join(vals)
        return ""


class Chapter:
    def __init__(self, title: str, items: list[epub.EpubItem]):
        self.title = title

        self._doc: lxml.html.HtmlElement = lxml.html.document_fromstring(
            items[0].get_content()
        )
        for item in items[1:]:
            item_body = cast(
                lxml.html.HtmlElement, lxml.html.document_fromstring(item.get_content())
            ).body
            self._doc.body.append(item_body)

    def is_valid(self, min_length: int = 8) -> bool:
        return len(self.blocks) >= min_length

    def append(self, text, tagname="p"):
        new_el = lxml.html.Element(tagname)
        new_el.text = text
        self._doc.body.append(new_el)

    def insert(self, pos, text, tagname="p"):
        new_el = lxml.html.Element(tagname)
        new_el.text = text
        self._doc.body.insert(pos, new_el)

    @property
    def blocks(self, match_tags: list[str] = ["p", "h1", "h2", "h3"], strip_empty=True):
        return [
            txt
            for tag in self._doc.iter(tag=match_tags)
            if (txt := tag.text_content()) or not strip_empty
        ]

    @property
    def elements(
        self, match_tags: list[str] = ["p", "h1", "h2", "h3"], strip_empty=False
    ):
        return [
            tag
            for tag in self._doc.iter(tag=match_tags)
            if tag.text_content() or not strip_empty
        ]

    def fulltext(self, delimiter: str = ""):
        return delimiter.join(self.blocks)

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"{self.title} [{len(self.blocks)}]"
