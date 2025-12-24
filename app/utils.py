from typing import Any, Iterable


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
