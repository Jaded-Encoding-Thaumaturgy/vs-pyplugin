from __future__ import annotations
from enum import IntEnum

from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar, cast

__all__ = [
    'SupportsKeysAndGetItem',
    'F',
    'copy_signature',
    'FilterMode'
]

_KT = TypeVar('_KT')
_VT_co = TypeVar('_VT_co', covariant=True)


class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]:
        ...

    def __getitem__(self, __k: _KT) -> _VT_co:
        ...


F = TypeVar('F', bound=Callable[..., Any])


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
