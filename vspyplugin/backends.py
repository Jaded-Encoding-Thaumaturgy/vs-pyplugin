from __future__ import annotations

from vstools import CustomIntEnum

__all__ = [
    'PyBackend'
]


class PyBackend(CustomIntEnum):
    NONE = -1
    NUMPY = 0
    CUPY = 1
    CUDA = 2
    CYTHON = 3

    def set_available(self, is_available: bool, e: ModuleNotFoundError | None = None) -> None:
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
    def import_error(self) -> ModuleNotFoundError | None:
        return next((e for backend, e in _unavailable_backends if backend is self), None)

    @property
    def dependencies(self) -> dict[str, str] | None:
        return _dependecies_backends.get(self, None)

    def set_dependencies(self, deps: dict[str, str]) -> None:
        _dependecies_backends[self] = {**deps}


_unavailable_backends = set[tuple[PyBackend, ModuleNotFoundError | None]]()
_dependecies_backends = dict[PyBackend, dict[str, str]]()
