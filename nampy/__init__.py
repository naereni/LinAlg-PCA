from .base import array, array
from ._linalg import T, det, inv, diag, trace, rank, norm, std
from . import utils, linalg, metrics
import types

array.T = property(T)
array.det = property(det)
array.inv = property(inv)
array.rank = property(rank)
array.trace = property(trace)
array.diag = property(diag)
array.norm = property(norm)
array.std = property(std)

_utils_exports = [
    name
    for name in dir(utils)
    if not name.startswith("_")
    and isinstance(
        getattr(utils, name),
        (types.FunctionType, int, float, str, list, dict, set, tuple),
    )
]

globals().update({name: getattr(utils, name) for name in _utils_exports})


__all__ = ["Matrix", "array"] + _utils_exports
