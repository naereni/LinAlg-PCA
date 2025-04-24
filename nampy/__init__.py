from .base import Matrix, array
from . import utils
import types

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
