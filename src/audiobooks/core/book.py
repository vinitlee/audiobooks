from __future__ import annotations
from typing import TYPE_CHECKING, cast

# if TYPE_CHECKING:

from typing import Any, Callable, Optional, Dict, Literal, Sequence

from pathlib import Path
import glob
import re
import warnings

from itertools import chain

# import json
# import yaml

import numpy as np
import cv2

from ebooklib import epub
import ebooklib
import lxml.html

from dataclasses import dataclass

from audiobooks.utils.utils import flatten_deep


class Book:
    epub: epub.EpubBook
    epub_path: Path
    meta: BookMetadata
    chapters: list[Chapter]

    def __init__(
        self,
        epub_path: Path | str,
    ):
        self.epub_path = Path(epub_path)
        self.epub = epub.read_epub(epub_path)

        self.meta = BookMetadata(self.epub)

        # self.generate_chapters()

    def generate_chapters(self):
        # Get Spine and TOC in IDs
        id_spine = [
            cast(epub.EpubItem, it).get_id()
            for iid, _ in self.epub.spine
            if (it := self.epub.get_item_with_id(iid)) is not None
        ]
        iter_toc = cast(list[epub.EpubItem], list(flatten_deep(self.epub.toc)))
        flat_toc = [
            toc_item for toc_item in iter_toc if isinstance(toc_item, epub.Link)
        ]
        id_toc = [
            cast(epub.EpubItem, it).get_id()
            for entry in flat_toc
            if (it := self.epub.get_item_with_href(entry.href.split("#")[0]))
            is not None
        ]
        # Also get TOC titles for later
        title_toc = [
            entry.title
            for entry in flat_toc
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
            for idx, (title, item_group) in enumerate(zip(title_toc, item_groups))
            if (ch := Chapter(title, idx, item_group)).is_valid()
        ]

        self.chapters[0].insert(0, f"{self.meta.title} by {self.meta.author}", "h1")

    def fulltext(self, block_delimiter: str = "\n", chapter_delimiter: str = "\n---\n"):
        return chapter_delimiter.join(
            [c.fulltext(block_delimiter) for c in self.chapters]
        )

    def __repr__(self) -> str:
        return f"{self.meta.title}: {len(self.chapters)} chapters"


class BookMetadata:
    epub_obj: epub.EpubBook
    epub_dc: dict
    cover: np.ndarray
    overrides: Dict[str, str]
    _delim: str = ", "
    _tags: list

    MetadataSource = Literal["override", "custom", "epub", "epub_meta", "fallback"]

    def __init__(self, epub_obj: epub.EpubBook):
        # Esential attributes
        self.epub_obj = epub_obj
        self.epub_dc = epub_obj.metadata[epub.NAMESPACES["DC"]]
        self.overrides = {}

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
            warnings.warn(
                "Could not get cover image. Defaulting to first image I find."
            )
            images = list(epub_obj.get_items_of_type(ebooklib.ITEM_IMAGE))
            if images:
                self.cover = cv2.imdecode(
                    np.frombuffer(images[0].content, np.uint8),
                    cv2.IMREAD_COLOR,
                )
            else:
                warnings.warn(
                    "Could not get any image. Giving up and providing a black square."
                )
                self.cover = np.zeros((512, 512, 3), dtype=np.uint8)

        # Inferred tags
        self._tags = list(self.epub_dc.keys())

    def get_from_epub_meta(self, name: str):
        entry = self.epub_dc.get(name)
        if entry is None:
            return None
        values, ids = list(zip(*entry))
        ids = [v.get("id") for v in ids]
        return self.join_vals(values, self._delim)

    @staticmethod
    def join_vals(vals: Sequence, delimiter: str):
        if len(vals) == 0:
            return None
        elif len(vals) == 1:
            return vals[0]

        return delimiter.join(vals)

    def in_epub(self, name: str):
        return name in self.epub_dc

    # One-off cases here
    @property
    def year(self):
        if self.source("year") in ["fallback"]:
            rights_str = str(self.get_tag("rights"))
            m = re.search(r"[0-9]{4}", rights_str)
            if m:
                return m.group(0)
            return ""
        else:
            return self.get_tag("")

    def get_tag(self, name):
        match self.source(name):
            case "override":
                return self.overrides[name]
            case "epub":
                return getattr(self.epub_obj, name)
            case "epub_meta":
                return self.get_from_epub_meta(name)
            case "fallback":
                return ""

    def source(self, name: str) -> MetadataSource:
        if name in self.overrides:
            return "override"
        elif name in ["title", "language"]:
            return "epub"
        elif self.in_epub(name):
            return "epub_meta"
        return "fallback"

    def __getattr__(self, name: str):
        # Handles general cases
        return self.get_tag(name)

    def add_overrides(self, **kwargs):
        self.overrides.update(kwargs)

    def __repr__(self) -> str:
        return f"{self.title}\n{self.overrides}"


class ElementBlock:
    element: lxml.html.HtmlElement
    tag: str
    text: str

    def __init__(self, element: lxml.html.HtmlElement):
        self.element = element
        self.tag = str(element.tag)
        self.text = element.text_content()

    def sanitize(self):
        self.text


class Chapter:
    _doc: lxml.html.HtmlElement
    index: int

    def __init__(self, title: str, index: int, items: list[epub.EpubItem]):
        self.title = title
        self.index = index

        self._doc = lxml.html.HtmlElement()

        if len(items):
            self._doc = lxml.html.document_fromstring(items[0].get_content())
            for item in items[1:]:
                item_body = cast(
                    lxml.html.HtmlElement,
                    lxml.html.document_fromstring(item.get_content()),
                ).body
                self._doc.body.append(item_body)

    def is_valid(self, min_length: int = 8) -> bool:
        return len(self.elements) >= min_length

    def append(self, text, tagname="p"):
        new_el = lxml.html.Element(tagname)
        new_el.text = text
        new_el.tag
        self._doc.body.append(new_el)

    def insert(self, pos, text, tagname="p"):
        new_el = lxml.html.Element(tagname)
        new_el.text = text
        self._doc.body.insert(pos, new_el)

    @property
    def blocks(self) -> list[str]:
        return [el.text for el in self.elements]

    @property
    def elements(self) -> list[ElementBlock]:
        match_tags: list[str] = ["p", "h1", "h2", "h3"]
        strip_empty = False

        return [
            ElementBlock(tag)
            for tag in self._doc.iter(tag=match_tags)
            if tag.text_content() or not strip_empty
        ]

    def fulltext(self, delimiter: str = ""):
        return delimiter.join(self.blocks)

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"{self.title} [{len(self.blocks)}]"
