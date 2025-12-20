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
        self.chapters = []
        for entry in self.epub.toc:
            title = entry.title
            item = self.epub.get_item_with_href(entry.href.split("#")[0])

            chapter = Chapter(title, cast(epub.EpubItem, item))
            if chapter.is_valid():
                self.chapters.append(chapter)
        # Hack to get the title at the beginning
        self.chapters[0].blocks.insert(
            0,
            f"{self.meta.title} by {self.meta.creator}",
        )

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
    def __init__(self, title: str, item: epub.EpubItem):
        self.title = title
        self.blocks: list[str]
        self.init_blocks(item)

    def is_valid(self, min_length: int = 8) -> bool:
        return len(self.blocks) >= min_length

    def init_blocks(self, epub_chapter: epub.EpubItem):
        content = epub_chapter.get_content()
        doc = lxml.html.document_fromstring(content)
        # Dump text of each <p> into its own block
        self.blocks = [tag.text_content() for tag in doc.iter(tag="p")]
        # Strip out empty blocks
        self.blocks = [t for t in self.blocks if len(t)]

    def fulltext(self, delimiter: str = ""):
        return delimiter.join(self.blocks)

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"{self.title} [{len(self.blocks)}]"
