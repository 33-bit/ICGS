from collections import OrderedDict
from copy import copy

import torch


def remove_prefix(text, prefix):
    """Remove one literal leading prefix fragment, if present."""
    return text[len(prefix):] if text.startswith(prefix) else text


def _remove_compiled_segments(key):
    return ".".join(segment for segment in key.split(".") if segment != "_orig_mod")


def _equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    try:
        result = left == right
        return bool(result) if not isinstance(result, torch.Tensor) else bool(result.all())
    except (RuntimeError, TypeError, ValueError):
        return False


def repair_checkpoint(path, save_path=None):
    """Write a repaired copy of a *trusted* Lightning checkpoint when needed.

    Loading with ``weights_only=False`` may execute pickle payloads. Only pass a
    checkpoint whose source you trust. ``save_path`` is explicit and may equal
    ``path`` for compatibility with the historical compiled-save call site.
    """
    ckpt = torch.load(path, weights_only=False)
    in_state_dict = ckpt["state_dict"]
    pairings = [
        (src_key, _remove_compiled_segments(src_key))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    if save_path is None:
        raise ValueError("save_path is required when checkpoint repair is needed")
    out_state_dict = OrderedDict()
    for src_key, dest_key in pairings:
        if dest_key in out_state_dict:
            if _equal(out_state_dict[dest_key], in_state_dict[src_key]):
                continue
            raise ValueError(
                f"checkpoint key collision after repair: {src_key!r} -> {dest_key!r}"
            )
        out_state_dict[dest_key] = in_state_dict[src_key]
    repaired = copy(ckpt)
    repaired["state_dict"] = out_state_dict
    torch.save(repaired, save_path)
