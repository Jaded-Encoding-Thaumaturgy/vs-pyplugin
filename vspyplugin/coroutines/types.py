"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, TypeVar

import vapoursynth as vs


class FrameRequest:
    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        raise NotImplementedError


T = TypeVar('T')
S = TypeVar('S')

AnyCoroutine = Coroutine[FrameRequest, S | None, T]
