from typing import Any, Iterable, Optional, TypeVar, Dict, Callable
from pathlib import Path

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def not_none_dict(d: Dict[K, V | None]) -> Dict[K, V]:
    """
    Strip None values from dictionary.

    :param d: Input dictionary
    :type d: dict
    """
    return {k: v for k, v in d.items() if v is not None}


def clean_dict(d: Dict[K, V]) -> Dict[K, V]:
    """
    Strip logical False values from dictionary.

    :param d: Input dictionary
    :type d: dict
    """
    return {k: v for k, v in d.items() if v}


def filter_dict(
    d: dict,
    whitelist: Optional[list[Any]] = None,
    blacklist: Optional[list[Any]] = None,
):
    def is_valid(k, v):
        non_null = v is not None
        white = True
        if whitelist is not None:
            white = k in whitelist
        black = True
        if blacklist is not None:
            black = k not in blacklist
        return non_null and white and black

    return {k: v for k, v in d.items() if is_valid(k, v)}


def maybe_parse(val: Any, parser: Callable[[Any], T]) -> Optional[T]:
    if val is None:
        return None
    return parser(val)


def maybe_dump(val: Any, dumper: Callable[[T], Any]) -> Any:
    if val is None:
        return None
    return dumper(val)


def ensure_list(e: Any):
    """
    Package non-list values as a list

    :param e: Value to evaluate
    :type e: Any
    """
    if not isinstance(e, list):
        return [e]
    return e


def flatten_deep(nested_iterable):
    for item in nested_iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_deep(item)
        else:
            yield item


def maybe_path(v: Optional[Path | str]) -> Optional[Path]:
    return Path(v) if v is not None else None
