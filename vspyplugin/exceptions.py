from __future__ import annotations

import textwrap
from traceback import format_exception
from typing import Any

from .base import PyBackend, PyPlugin


class FormattedException(Exception):
    def __init__(self, e: Exception) -> None:
        formatted = ''.join(
            format_exception(type(e), e, e.__traceback__)
        )
        formatted = textwrap.indent(formatted, '| ')
        super().__init__(f'Something went wrong.\n{formatted}')


class UnavailableBackend(ValueError):
    """Raised when trying to initialize an unavailable backend"""

    def __init__(
        self, backend: PyBackend, _class: PyPlugin[Any],
        message: str = '{class_name}: This plugin is built on top of the {backend} backend which is unavailable!'
    ) -> None:
        self.backend = backend._name_
        self.class_name = _class.__class__.__name__
        self.message: str = message

        super().__init__(self.message.format(class_name=self.class_name, backend=self.backend))
