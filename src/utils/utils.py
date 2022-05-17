from collections.abc import MutableMapping


def flatten_dict(d: MutableMapping, parent_key: str = None, sep: str = '.') -> MutableMapping:
    """flatten multi-level dictionary

    Args:
        d (MutableMapping): multi-level dictionary
        parent_key (str, optional): prefix. Defaults to ''.
        sep (str, optional): separator between parents-level and child. Defaults to '.'.

    Returns:
        dict: flat dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
