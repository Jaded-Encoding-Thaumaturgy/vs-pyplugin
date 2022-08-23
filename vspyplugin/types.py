from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar, Union, cast

import vapoursynth as vs

__all__ = [
    'SupportsKeysAndGetItem',
    'F', 'F_VD',
    'FD_T', 'DT_T',
    'copy_signature',
    'FilterMode',
    'OutputFunc_T'
]

_KT = TypeVar('_KT')
_VT_co = TypeVar('_VT_co', covariant=True)


F = TypeVar('F', bound=Callable[..., Any])
F_VD = TypeVar('F_VD', bound=Callable[..., vs.VideoNode])


class SupportsIndexing(Protocol[_VT_co]):
    def __getitem__(self, __k: int) -> _VT_co:
        ...


class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]:
        ...

    def __getitem__(self, __k: _KT) -> _VT_co:
        ...


class copy_signature(Generic[F]):
    def __init__(self, target: F) -> None:
        ...

    def __call__(self, wrapped: Callable[..., Any]) -> F:
        return cast(F, wrapped)


class FilterMode(IntEnum):
    Serial = 0
    """Serial processing"""

    Parallel = 1
    """Parallel requests"""

    Async = 2
    """Async and parallelized requests"""


OutputFunc_T = Union[
    Callable[[vs.VideoFrame, int], vs.VideoFrame], Callable[[tuple[vs.VideoFrame, ...], int], vs.VideoFrame]
]

FD_T = TypeVar('FD_T', bound=Any | SupportsKeysAndGetItem[str, object] | None)
DT_T = TypeVar('DT_T')

PassthroughC = Callable[[F], F]
