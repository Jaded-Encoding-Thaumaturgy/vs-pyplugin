from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar

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
        ...
