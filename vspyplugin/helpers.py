"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Generator, Generic, TypeVar
import vapoursynth as vs

from .exceptions import FormattedException

core = vs.core


UNWRAP_NAME = '__vspyplugin_unwrap'


class FrameRequest:
    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        raise NotImplementedError


FrameCoroutine = Coroutine[FrameRequest, Any, vs.VideoFrame]

T = TypeVar('T')

AnyCoroutine = Coroutine[FrameRequest, Any, T]

F = TypeVar('F', bound=Callable[..., FrameCoroutine])


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
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        req_clip = self.clip[self.frame_no] * (frame_no + 1)

        return clip.std.FrameEval(
            lambda n, f: continuation(f), [req_clip]
        )


class GatherRequests(Generic[T], FrameRequest):
    def __init__(self, coroutines: tuple[AnyCoroutine[T], ...]) -> None:
        self.coroutines = coroutines

    def __await__(self) -> Generator[GatherRequests[T], None, tuple[T, ...]]:
        return (yield self)  # type: ignore

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        wrapped = [
            _coro2node_wrapped(clip, frame_no, coro)
            for coro in self.coroutines
        ]

        def _apply(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
            return continuation(
                tuple(
                    _unwrap(fr, wrapped[fn][1])
                    for fn, fr in enumerate(f)
                )
            )
        return clip.std.FrameEval(_apply, prop_src=[c for c, _ in wrapped])


async def get_frame(clip: vs.VideoNode, frame_no: int) -> vs.VideoFrame:
    return await SingleFrameRequest(clip, frame_no)


async def gather(*coroutines: AnyCoroutine[T]) -> tuple[T, ...]:
    return await GatherRequests(coroutines)


def _unwrap(frame: vs.VideoFrame, atom: Atom[Any]) -> Any:
    if frame.props.get(UNWRAP_NAME, False):
        return atom.value
    else:
        return frame


def _coro2node_wrapped(
    base_clip: vs.VideoNode, frameno: int, coro: AnyCoroutine[T]
) -> tuple[vs.VideoNode, Atom[T]]:
    atom = Atom[T]()
    return _coro2node(base_clip, frameno, coro, atom), atom  # type: ignore


def _coro2node(
    base_clip: vs.VideoNode, frameno: int, coro: FrameCoroutine, wrap: Atom[Any] | None = None
) -> vs.VideoNode:
    assert base_clip.format

    bc = core.std.BlankClip(
        width=base_clip.width, height=base_clip.height, keep=True,
        format=base_clip.format.id, length=1, fpsnum=1, fpsden=1
    )

    props_clip = base_clip.std.BlankClip()

    def _continue(value: Any) -> vs.VideoNode:
        if wrap:
            wrap.unset()

        try:
            next_request = coro.send(value)
        except StopIteration as e:
            if isinstance(e.value, vs.VideoNode):
                return e.value
            elif isinstance(e.value, vs.VideoFrame):
                frame = e.value.copy()
                return bc.std.ModifyFrame(bc, lambda n, f: frame).std.Loop(frameno + 1)
            elif not wrap:
                raise ValueError("You can only return a Frames and VideoNodes here.")
            else:
                wrap.set(e.value)
                return props_clip.std.SetFrameProp(UNWRAP_NAME, intval=True)
        except Exception as e:
            raise FormattedException(e)
        else:
            return next_request.build_frame_eval(base_clip, frameno, _continue)
    return _continue(None)


def frame_eval(base_clip: vs.VideoNode) -> Callable[[F], vs.VideoNode]:
    def _decorator(func: F) -> vs.VideoNode:
        return base_clip.std.FrameEval(
            lambda n: _coro2node(base_clip, n, func(n))
        )
    return _decorator
