"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Callable, Literal, overload

import vapoursynth as vs
from vstools import CustomValueError

from .coros import coro2node
from .types import AnyCoroutine

__all__ = [
    'frame_eval', 'frame_eval_async'
]

core = vs.core


FE_N_FUNC = Callable[[int], vs.VideoNode]
FE_F_FUNC = Callable[[int, vs.VideoFrame], vs.VideoNode]
FE_L_FUNC = Callable[[int, list[vs.VideoFrame]], vs.VideoNode]

FEA_FUNC = Callable[[int], AnyCoroutine[None, vs.VideoFrame | vs.VideoNode]]


def frame_eval_async(base_clip: vs.VideoNode) -> Callable[[FEA_FUNC], vs.VideoNode]:
    def _decorator(func: FEA_FUNC) -> vs.VideoNode:
        def _inner(n: int) -> vs.VideoNode:
            return coro2node(base_clip, n, func(n))

        return base_clip.std.FrameEval(_inner)

    return _decorator


@overload
def frame_eval(base_clip: vs.VideoNode, /) -> Callable[[FE_N_FUNC], vs.VideoNode]:  # type: ignore
    ...


@overload
def frame_eval(base_clip: vs.VideoNode, /, frame: Literal[True] = ...) -> Callable[[FE_F_FUNC], vs.VideoNode]:
    ...


@overload
def frame_eval(  # type: ignore
    base_clip: vs.VideoNode, frame_clips: vs.VideoNode = ..., /, frame: bool = ...
) -> Callable[[FE_F_FUNC], vs.VideoNode]:
    ...


@overload
def frame_eval(  # type: ignore
    base_clip: vs.VideoNode, frame_clips: None = ..., /, frame: Literal[True] = ...
) -> Callable[[FE_F_FUNC], vs.VideoNode]:
    ...


@overload
def frame_eval(
    base_clip: vs.VideoNode, frame_clips: list[vs.VideoNode] = ..., /, frame: bool = ...
) -> Callable[[FE_L_FUNC], vs.VideoNode]:
    ...


def frame_eval(  # type: ignore
    base_clip: vs.VideoNode, frame_clips: vs.VideoNode | list[vs.VideoNode] | None = None, /, frame: bool = False
) -> Callable[[FE_N_FUNC | FE_F_FUNC | FE_L_FUNC], vs.VideoNode]:
    if frame and not frame_clips:
        frame_clips = base_clip

    def _decorator(func: FE_N_FUNC | FE_F_FUNC | FE_L_FUNC) -> vs.VideoNode:
        args = func.__annotations__
        keys = list(filter(lambda x: x not in {'self', 'return'}, args.keys()))

        n_args = len(keys)

        if n_args == 1:
            if 'n' in keys:
                _inner = func
            else:
                def _inner(n: int) -> vs.VideoNode:
                    return func(n)  # type: ignore
        elif n_args == 2:
            if isinstance(frame_clips, list) and len(frame_clips) < 2:
                def _inner(n: int, f: vs.VideoFrame) -> vs.VideoNode:
                    return func(n, [f])  # type: ignore
            else:
                if 'n' in keys and 'f' in keys:
                    _inner = func
                else:
                    def _inner(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
                        return func(n, f)  # type: ignore
        else:
            raise CustomValueError('Function must have 1-2 arguments!', frame_eval)

        return base_clip.std.FrameEval(_inner, frame_clips, base_clip)

    return _decorator
