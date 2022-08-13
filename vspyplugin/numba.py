from __future__ import annotations

from .base import FD_T, PyBackend, PyPlugin, PyPluginUnavailableBackend

__all__ = [
    'PyPluginNumba'
]

this_backend = PyBackend.NUMBA

try:
    class PyPluginNumba(PyPlugin[FD_T]):
        backend = this_backend

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginNumba(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
