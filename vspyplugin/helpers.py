"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Generator, Generic, Iterable, Literal, TypeVar, overload

import vapoursynth as vs

from .exceptions import FormattedException

core = vs.core

__all__ = [
    'FrameRequest', 'SingleFrameRequest', 'GatherRequests',
    'Atom',

    'get_frame', 'get_frames', 'get_frames_shifted',
    'gather', 'gathers',

    'frame_eval', 'frame_eval_async',

    'AnyCoroutine'
]

UNWRAP_NAME = '__vspyplugin_unwrap'


class FrameRequest:
    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        raise NotImplementedError


T = TypeVar('T')
S = TypeVar('S')

AnyCoroutine = Coroutine[FrameRequest, S | None, T]


FE_N_FUNC = Callable[[int], vs.VideoNode]
FE_F_FUNC = Callable[[int, vs.VideoFrame], vs.VideoNode]
FE_L_FUNC = Callable[[int, list[vs.VideoFrame]], vs.VideoNode]

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


async def get_frames(*clips: vs.VideoNode, frame_no: int) -> tuple[vs.VideoFrame, ...]:
    return await gathers(get_frame(clip, frame_no) for clip in clips)


async def get_frames_shifted(
    clip: vs.VideoNode, frame_no: int, shifts: int | tuple[int, int] | Iterable[int] = (-1, 1)
) -> tuple[vs.VideoFrame, ...]:
    if isinstance(shifts, int):
        shifts = (-shifts, shifts)

    if isinstance(shifts, tuple):
        start, stop = shifts
        step = -1 if stop < start else 1
        shifts = range(start, stop + step, step)

    return await gathers(get_frame(clip, frame_no + shift) for shift in shifts)


async def gather(*coroutines: AnyCoroutine[S, T]) -> tuple[T, ...]:
    return await GatherRequests(coroutines)


async def gathers(coroutines: Iterable[AnyCoroutine[S, T]]) -> tuple[T, ...]:
    return await GatherRequests(tuple(coroutines))


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
            raise ValueError('frame_eval: Function must have 1-2 arguments!')

        return base_clip.std.FrameEval(_inner, frame_clips, base_clip)

    return _decorator
