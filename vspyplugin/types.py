from __future__ import annotations

from typing import Any, Callable, TypeVar, Union

from vstools import CustomIntEnum, SupportsKeysAndGetItem, vs

__all__ = [
    'FilterMode',

    'FD_T', 'DT_T',
    'OutputFunc_T'
]


class FilterMode(CustomIntEnum):
    Serial = 0
    """Serial processing"""

    Parallel = 1
    """Parallel requests"""

    Async = 2
    """Async and parallelized requests"""


FD_T = TypeVar('FD_T', bound=Any | SupportsKeysAndGetItem[str, object] | None)
DT_T = TypeVar('DT_T')

OutputFunc_T = Union[
    Callable[[vs.VideoFrame, int], vs.VideoFrame], Callable[[tuple[vs.VideoFrame, ...], int], vs.VideoFrame]
]
