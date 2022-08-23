from __future__ import annotations

from enum import IntEnum

__all__ = [
    'PyBackend'
]


class PyBackend(IntEnum):
    NONE = -1
    NUMPY = 0
    CUPY = 1
    CUDA = 2
    CYTHON = 3

    def set_available(self, is_available: bool, e: BaseException | None = None) -> None:
        if not is_available:
            _unavailable_backends.add((self, e))
        elif not self.is_available:
            unav_backs = _unavailable_backends.copy()
            _unavailable_backends.clear()
            _unavailable_backends.update({
                (backend, error) for backend, error in unav_backs if backend is not self
            })

    @property
    def is_available(self) -> bool:
        return self not in {backend for backend, _ in _unavailable_backends}

    @property
    def import_error(self) -> BaseException | None:
        return next((e for backend, e in _unavailable_backends if backend is self), None)


_unavailable_backends = set[tuple[PyBackend, BaseException | None]]()
