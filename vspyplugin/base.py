from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
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

    @abstractmethod
    def eval_single_clip_per_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
        ...

    @abstractmethod
    def eval_single_clip_one_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
        ...

    @abstractmethod
    def eval_multi_clips_per_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
        ...

    @abstractmethod
    def eval_multi_clips_one_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
        ...

    @abstractmethod
    def to_host(self, f: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
        ...

    @abstractmethod
    def from_host(self, src: Any, dst: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
        ...

    def __class_getitem__(cls, fdata: Type[FD_T] | None = None) -> Type[PyPlugin[FD_T]]:
        if fdata is None:
            fdata = GenericFilterData  # type: ignore

        class PyPluginInnerClass(cls):  # type: ignore
            filter_data = fdata

        return PyPluginInnerClass

    def get_filter_data(self, **kwargs: Any) -> FD_T:
        return self.filter_data(**kwargs)

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

        self.fd = self.get_filter_data(**kwargs)

    def invoke(self, **kwargs: Any) -> vs.VideoNode:
        class_name = self.__class__.__name__

        if kwargs:
            self.fd = self.get_filter_data(**kwargs)

        n_clips = len(self.clips)

        if n_clips < self.min_clips or (self.max_clips > 0 and n_clips > self.max_clips):
            max_clips = 'inf' if self.max_clips == -1 else self.max_clips
            raise ValueError(
                f'{class_name}: You must pass {self.min_clips} <= n clips <= {max_clips}!'
            )

        if n_clips == self.omit_first_clip == 1:
            raise ValueError(
                f'{class_name}: You must have at least two clips to omit the first clip!'
            )

        if self.out_format.num_planes > 1 or self.out_format.subsampling_w or self.out_format.subsampling_h:
            function = self.eval_single_clip_per_plane if n_clips == 1 else self.eval_multi_clips_per_plane
        else:
            function = self.eval_single_clip_one_plane if n_clips == 1 else self.eval_multi_clips_one_plane

        return self._post_invoke(cast(Callable[..., vs.VideoFrame], function))

    def _post_invoke(self, function: Callable[..., vs.VideoFrame]) -> vs.VideoNode:
        assert self.ref_clip.format

        out = self.ref_clip.std.ModifyFrame(self.clips, function)

        if self.out_format.id != self.ref_clip.format.id:
            return out.resize.Bicubic(format=self.out_format.id, dither_type='none')

        return out


class PyPluginUnavailableBackend(PyPlugin[FD_T]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from .exceptions import UnavailableBackend

        raise UnavailableBackend(self.backend, self)
