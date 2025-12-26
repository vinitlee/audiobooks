from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    from kokoro import KPipeline, KModel
    from misaki.en import G2P, MToken

from typing import Callable, Tuple, List
import logging
import warnings

from functools import lru_cache

DEFAULT_KOKORO_REPO = "hexgrad/Kokoro-82M"


class KPipelineLazy:
    g2p: Callable[[str], Tuple[str, List[MToken]]]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._obj = None

    def load(self):
        if self._obj is None:
            from kokoro import KPipeline

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                self._obj = KPipeline(*self.args, **self.kwargs)
        return self._obj

    @classmethod
    @lru_cache(maxsize=1)
    def instance(cls, *args, **kwargs) -> KPipelineLazy:
        return cls(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)
