from __future__ import annotations

from typing import Any

from .backends import PyBackend
from .base import PyPluginBase


class UnavailableBackend(ValueError):
    """Raised when trying to initialize an unavailable backend"""

    def __init__(
        self, backend: PyBackend, _class: PyPluginBase[Any, Any],
        message: str = '{class_name}: This plugin is built on top of the {backend} backend which is unavailable!'
    ) -> None:
        self.backend = backend._name_
        self.class_name = _class.__class__.__name__
        self.message: str = message

        super().__init__(self.message.format(class_name=self.class_name, backend=self.backend))
