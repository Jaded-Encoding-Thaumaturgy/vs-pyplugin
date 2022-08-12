from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Generic, Literal, Type, TypeVar, cast, overload

import vapoursynth as vs

__all__ = [
    'PyBackend', 'PyPlugin',
    'GenericFilterData',
    'FD_T',
    'PyPluginUnavailableBackend'
]


class PyBackend(IntEnum):
    NUMPY = 0
    CUPY = 1
    CYTHON = 2
    CUDA = 3

    def set_available(self, is_available: bool, e: BaseException | None = None) -> None:
        if not is_available:
            _unavailable_backends.add((self, e))
        elif not self.is_available:
            unav_backs = _unavailable_backends.copy()
            _unavailable_backends.clear()
            _unavailable_backends.update({
                (backend, error) for backend, error in unav_backs if backend is not self
            })

    @property
    def is_available(self) -> bool:
        return self not in {backend for backend, _ in _unavailable_backends}

    @property
    def import_error(self) -> BaseException | None:
        return next((e for backend, e in _unavailable_backends if backend is self), None)


_unavailable_backends = set[tuple[PyBackend, BaseException | None]]()

FD_T = TypeVar('FD_T')
F = TypeVar('F', bound=Callable[..., vs.VideoNode])


class GenericFilterData(dict[str, Any]):
    ...


class PyPlugin(Generic[FD_T]):
    __slots__ = (
        'backend', 'filter_data', 'clips', 'ref_clip', 'fd'
    )

    backend: PyBackend

    filter_data: Type[FD_T]

    # Implementation configs
    min_clips: int = 1
    max_clips: int = -1
    omit_first_clip: bool = False
    channels_last: bool = True

    clips: list[vs.VideoNode]
    ref_clip: vs.VideoNode
    out_format: vs.VideoFormat

    fd: FD_T

    float_processing: bool | Literal[16, 32] = False

    @staticmethod
    def ensure_output(func: F) -> F:
        @wraps(func)
        def _wrapper(self: PyPlugin[FD_T], *args: Any, **kwargs: Any) -> Any:
            assert self.ref_clip.format

            out = func(self, *args, **kwargs)

            if self.out_format.id != self.ref_clip.format.id:
                return out.resize.Bicubic(format=self.out_format.id, dither_type='none')

            return out

        return cast(F, _wrapper)

    @abstractmethod
    def to_host(self, f: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
        ...

    @abstractmethod
    def from_host(self, src: Any, dst: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
        ...

    def process(self, src: Any, dst: Any, n: int) -> None:
        raise NotImplementedError

    def __class_getitem__(cls, fdata: Type[FD_T] | None = None) -> Type[PyPlugin[FD_T]]:
        if fdata is None:
            fdata = GenericFilterData  # type: ignore

        class PyPluginInnerClass(cls):  # type: ignore
            filter_data = fdata

        return PyPluginInnerClass

    @overload
    def norm_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...

    @overload
    def norm_clip(self, clip: None) -> None:
        ...

    def norm_clip(self, clip: vs.VideoNode | None) -> vs.VideoNode | None:
        if not clip:
            return clip

        assert clip.format

        if self.float_processing:
            bps = 32 if self.float_processing is True else self.float_processing

            if clip.format.sample_type is not vs.FLOAT or clip.format.bits_per_sample != bps:
                return clip.resize.Point(
                    format=clip.format.replace(sample_type=vs.FLOAT, bits_per_sample=bps).id,
                    dither_type='none'
                )

        return clip

    def __init__(
        self, clips: vs.VideoNode | list[vs.VideoNode], ref_clip: vs.VideoNode | None = None, **kwargs: Any
    ) -> None:
        clips = clips if isinstance(clips, list) else [clips]

        if self.filter_data is None or isinstance(self.filter_data, TypeVar):
            self.filter_data = GenericFilterData  # type: ignore

        rclip = ref_clip or clips[0]
        assert rclip.format

        self.out_format = rclip.format  # type: ignore

        self.clips = [self.norm_clip(clip) for clip in clips]
        self.ref_clip = self.norm_clip(ref_clip) if ref_clip else self.clips[0]

        self.fd = self.filter_data(**kwargs)

        n_clips = len(self.clips)

        class_name = self.__class__.__name__

        if n_clips < self.min_clips or (self.max_clips > 0 and n_clips > self.max_clips):
            max_clips = 'inf' if self.max_clips == -1 else self.max_clips
            raise ValueError(
                f'{class_name}: You must pass {self.min_clips} <= n clips <= {max_clips}!'
            )

        if n_clips == self.omit_first_clip == 1:
            raise ValueError(
                f'{class_name}: You must have at least two clips to omit the first clip!'
            )

    def invoke(self) -> vs.VideoNode:
        raise NotImplementedError


class PyPluginUnavailableBackend(PyPlugin[FD_T]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from .exceptions import UnavailableBackend

        raise UnavailableBackend(self.backend, self)
