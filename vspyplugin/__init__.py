from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    import sys
    from pathlib import Path

    if sys.orig_argv:
        path = Path(sys.orig_argv[max(0, min(len(sys.orig_argv) - 1, 1))])

        if Path(sys.executable).parent == path.parent.parent and path.name in [
            'vspyplugin-script.py', 'vspyplugin.exe', 'vspyplugin'
        ]:
            import os
            os.environ['vspyplugin_is_cli'] = 'True'

# ruff: noqa: F401, F403

from .abstracts import *
from .backends import *
from .base import *
from .coroutines import *
from .cuda import *
from .cupy import *
from .cython import *
from .exceptions import *
from .numpy import *
from .types import *
from .utils import *
