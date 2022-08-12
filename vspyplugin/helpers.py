"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Generator, Generic, Iterable, TypeVar

import vapoursynth as vs

from .exceptions import FormattedException

core = vs.core

UNWRAP_NAME = '__vspyplugin_unwrap'


class FrameRequest:
    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        raise NotImplementedError


T = TypeVar('T')
S = TypeVar('S')

AnyCoroutine = Coroutine[FrameRequest, S | None, T]

FEA_FUNC = Callable[[int], AnyCoroutine[None, vs.VideoFrame | vs.VideoNode]]


class Atom(Generic[T]):
    value: T | None

    def __init__(self) -> None:
        self.value = None

    def set(self, value: T) -> None:
        self.value = value

    def unset(self) -> None:
        self.value = None


class SingleFrameRequest(FrameRequest):
    def __init__(self, clip: vs.VideoNode, frame_no: int) -> None:
        self.clip = clip
        self.frame_no = frame_no

    def __await__(self) -> Generator[SingleFrameRequest, None, vs.VideoFrame]:
        return (yield self)  # type: ignore

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int,
        continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        req_clip = self.clip[self.frame_no]

        return clip.std.FrameEval(
            lambda n, f: continuation(f), [req_clip]
        )


class GatherRequests(Generic[T], FrameRequest):
    def __init__(self, coroutines: tuple[AnyCoroutine[S, T], ...]) -> None:
        if len(coroutines) <= 1:
            raise ValueError('GatherRequests: you need to pass at least 2 coroutines!')

        self.coroutines = coroutines

    def __await__(self) -> Generator[GatherRequests[T], None, tuple[T, ...]]:
        return (yield self)  # type: ignore

    @staticmethod
    def _unwrap(frame: vs.VideoFrame, atom: Atom[T]) -> vs.VideoFrame | T | None:
        if frame.props.get(UNWRAP_NAME, False):
            return atom.value

        return frame

    def unwrap_coros(self, clip: vs.VideoNode, frame_no: int) -> tuple[list[vs.VideoNode], list[Atom[T]]]:
        return zip(*[  # type: ignore
            _coro2node_wrapped(clip, frame_no, coro) for coro in self.coroutines
        ])

    def wrap_frames(self, frames: list[vs.VideoFrame], atoms: list[Atom[T]]) -> tuple[vs.VideoFrame | T | None, ...]:
        return tuple(
            self._unwrap(frame, atom) for frame, atom in zip(frames, atoms)
        )

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int,
        continuation: Callable[[tuple[vs.VideoFrame | T | None, ...]], vs.VideoNode]
    ) -> vs.VideoNode:
        clips, atoms = self.unwrap_coros(clip, frame_no)

        def _apply(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
            return continuation(self.wrap_frames(f, atoms))

        return clip.std.FrameEval(_apply, clips)


async def get_frame(clip: vs.VideoNode, frame_no: int) -> vs.VideoFrame:
    return await SingleFrameRequest(clip, frame_no)


async def get_frames(
    clip: vs.VideoNode, frame_no: int, shifts: int | tuple[int, int] | Iterable[int] = (-1, 1)
) -> tuple[vs.VideoFrame, ...]:
    if isinstance(shifts, int):
        shifts = (-shifts, shifts)

    if isinstance(shifts, tuple):
        start, stop = shifts
        step = -1 if stop < start else 1
        shifts = range(start, stop + step, step)

    coroutines = (
        get_frame(clip, frame_no + shift) for shift in shifts
    )

    return await gather(*coroutines)


async def gather(*coroutines: AnyCoroutine[S, T]) -> tuple[T, ...]:
    return await GatherRequests(coroutines)


def _wrapped_modify_frame(blank_clip: vs.VideoNode) -> Callable[[vs.VideoFrame], vs.VideoNode]:
    def _wrap_frame(frame: vs.VideoFrame) -> vs.VideoNode:
        def _return_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            return frame

        return blank_clip.std.ModifyFrame(blank_clip, _return_frame)

    return _wrap_frame


def _coro2node_wrapped(
    base_clip: vs.VideoNode, frameno: int, coro: AnyCoroutine[S, T]
) -> tuple[vs.VideoNode, Atom[T]]:
    atom = Atom[T]()
    return _coro2node(base_clip, frameno, coro, atom), atom


def _coro2node(
    base_clip: vs.VideoNode, frameno: int, coro: AnyCoroutine[S, T], wrap: Atom[T] | None = None
) -> vs.VideoNode:
    assert base_clip.format

    props_clip = base_clip.std.BlankClip()
    blank_clip = core.std.BlankClip(
        length=1, fpsnum=1, fpsden=1, keep=True,
        width=base_clip.width, height=base_clip.height,
        format=base_clip.format.id
    )

    _wrap_frame = _wrapped_modify_frame(blank_clip)

    def _continue(wrapped_value: S | None) -> vs.VideoNode:
        if wrap:
            wrap.unset()

        try:
            next_request = coro.send(wrapped_value)
        except StopIteration as e:
            value = e.value

            if isinstance(value, vs.VideoNode):
                return value

            if isinstance(value, vs.VideoFrame):
                return _wrap_frame(value)

            if not wrap:
                raise ValueError('frame_eval_async: You can only return a VideoFrame or VideoNode!')

            wrap.set(value)

            return props_clip.std.SetFrameProp(UNWRAP_NAME, intval=True)
        except Exception as e:
            raise FormattedException(e)

        return next_request.build_frame_eval(base_clip, frameno, _continue)

    return _continue(None)


def frame_eval_async(base_clip: vs.VideoNode) -> Callable[[FEA_FUNC], vs.VideoNode]:
    def _decorator(func: FEA_FUNC) -> vs.VideoNode:
        def _inner(n: int) -> vs.VideoNode:
            return _coro2node(base_clip, n, func(n))

        return base_clip.std.FrameEval(_inner)

    return _decorator
