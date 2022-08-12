"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

import typing as t

import vapoursynth as vs

from .exceptions import FormattedException

core = vs.core


UNWRAP_NAME = "__vspyplugin_unwrap"
T = t.TypeVar("T")


class Atom(t.Generic[T]):
    value: t.Optional[T]

    def __init__(self) -> None:
        self.value = None

    def set(self, value: T):
        self.value = value

    def unset(self):
        self.value = None


def frame2clip(frame: vs.VideoFrame, /) -> vs.VideoNode:
    """Converts a VapourSynth frame to a clip.
    :param frame:          The frame to convert.
    :param enforce_cache:  Unused, deprecated parameter. Kept for compatibility.
    :return: A one-frame clip that yields the `frame` passed to the function.
    """
    bc = core.std.BlankClip(
        width=frame.width,
        height=frame.height,
        length=1,
        fpsnum=1,
        fpsden=1,
        format=frame.format.id
    )
    frame = frame.copy()
    return bc.std.ModifyFrame([bc], lambda n, f: frame.copy())


class FrameRequest:
    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: t.Callable[[t.Any], vs.VideoNode]
    ) -> vs.VideoNode:
        raise NotImplementedError()


FrameCoroutine = t.Coroutine[FrameRequest, t.Any, vs.VideoFrame]
AnyCoroutine = t.Coroutine[FrameRequest, t.Any, T]


class SingleFrameRequest(FrameRequest):

    def __init__(self, clip: vs.VideoNode, frame_no: int) -> None:
        self.clip = clip
        self.frame_no = frame_no

    def __await__(self):
        return (yield self)

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: t.Callable[[t.Any], vs.VideoNode]
    ) -> vs.VideoNode:
        req_clip = self.clip[self.frame_no] * (frame_no + 1)

        def _apply(n, f):
            return continuation(f)
        return clip.std.FrameEval(_apply, prop_src=[req_clip])


class Gather(FrameRequest):
    def __init__(self, coros: t.List[FrameCoroutine]) -> None:
        self.coros = coros

    def __await__(self):
        return (yield self)

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: t.Callable[[t.Any], vs.VideoNode]
    ) -> vs.VideoNode:
        wrapped = [
            _coro2node_wrapped(clip, frame_no, coro)
            for coro in self.coros
        ]

        def _apply(n, f):
            return continuation(tuple(
                _unwrap(fr, wrapped[fn][1])
                for fn, fr in enumerate(f))
            )
        return clip.std.FrameEval(_apply, prop_src=[c for c, _ in wrapped])


async def get_frame(clip: vs.VideoNode, frame_no: int) -> vs.VideoFrame:
    return await SingleFrameRequest(clip, frame_no)


async def gather(*coros: AnyCoroutine[t.Any]) -> t.Tuple[t.Any, ...]:
    return await Gather(list(coros))


def _unwrap(frame: vs.VideoFrame, atom: Atom[t.Any]) -> t.Any:
    if frame.props.get(UNWRAP_NAME, False):
        return atom.value
    else:
        return frame


def _coro2node_wrapped(base_clip: vs.VideoNode, frameno: int, coro: AnyCoroutine[T]) -> t.Tuple[vs.VideoNode, Atom[T]]:
    atom = Atom()
    return _coro2node(base_clip, frameno, coro, atom), atom


def _coro2node(
    base_clip: vs.VideoNode, frameno: int, coro: FrameCoroutine, wrap: t.Optional[Atom[t.Any]] = None
) -> vs.VideoNode:
    def _continue(value: t.Any) -> vs.VideoNode:
        if wrap:
            wrap.unset()
        try:
            next_request = coro.send(value)
        except StopIteration as e:
            if isinstance(e.value, vs.VideoNode):
                return e
            elif isinstance(e.value, vs.VideoFrame):
                return frame2clip(e.value) * (frameno + 1)
            elif not wrap:
                raise ValueError("You can only return a Frames and VideoNodes here.")
            else:
                wrap.set(e.value)
                return base_clip.std.BlankClip().std.SetFrameProp(UNWRAP_NAME, intval=True)

        except Exception as e:
            raise FormattedException(e)
        else:
            return next_request.build_frame_eval(
                base_clip,
                frameno,
                _continue
            )
    return _continue(None)


def video(base_clip: vs.VideoNode):
    def _decorator(func):
        def _enter(n):
            return _coro2node(base_clip, n, func(n))
        return base_clip.std.FrameEval(_enter)
    return _decorator
