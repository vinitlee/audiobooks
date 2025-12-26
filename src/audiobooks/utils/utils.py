from typing import Any, Iterable, Optional


def not_none_dict(d: dict):
    """
    Strip None values from dictionary.

    :param d: Input dictionary
    :type d: dict
    """
    return {k: v for k, v in d.items() if v is not None}


def clean_dict(d: dict):
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
