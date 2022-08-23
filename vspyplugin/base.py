from __future__ import annotations


from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Generic, Type, TypeVar, overload, cast

import vapoursynth as vs

from .abstracts import PyPluginBackendBase, FD_T, DT_T
from .backends import PyBackend
from .coroutines import frame_eval_async, get_frame, get_frames
from .types import FilterMode, copy_signature

__all__ = [
    'PyPlugin',
    'PyPluginUnavailableBackend'
]


@dataclass
class PyPluginOptions:
    force_precision: int | None = None
    shift_chroma: bool = False

    @overload
    def norm_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...

    @overload
    def norm_clip(self, clip: None) -> None:
        ...

    def norm_clip(self, clip: vs.VideoNode | None) -> vs.VideoNode | None:
        if not clip:
            return clip

        assert (fmt := clip.format)

        if self.force_precision:
            if fmt.sample_type is not vs.FLOAT or fmt.bits_per_sample != self.force_precision:
                clip = clip.resize.Point(
                    format=fmt.replace(sample_type=vs.FLOAT, bits_per_sample=self.force_precision).id,
                    dither_type='none'
                )

        if self.shift_chroma:
            if fmt.sample_type is not vs.FLOAT and self.force_precision != 32:
                raise ValueError(
                    f'{self.__class__.__name__}: You need to have a clip with float sample type for shift_chroma=True!'
                )

            if fmt.num_planes == 3:
                clip = clip.std.Expr(['', 'x 0.5 +'])

        return clip

    def ensure_output(self, plugin: PyPlugin[FD_T], clip: vs.VideoNode) -> vs.VideoNode:
        assert plugin.ref_clip.format

        if plugin.out_format.id != plugin.ref_clip.format.id:
            return clip.resize.Bicubic(format=plugin.out_format.id, dither_type='none')

        return clip


class PyPluginBase(Generic[FD_T, DT_T], PyPluginBackendBase[DT_T]):
    if TYPE_CHECKING:
        __slots__ = (
            'backend', 'filter_data', 'clips', 'ref_clip', 'fd',
            '_input_per_plane', 'out_format', 'output_per_plane',
            'is_single_plane'
        )
    else:
        __slots__ = (
            'backend', 'filter_data', 'clips', 'ref_clip', 'fd',
            '_input_per_plane'
        )

    backend: PyBackend
    filter_data: Type[FD_T]
    filter_mode: FilterMode

    options: PyPluginOptions

    input_per_plane: bool | list[bool]
    output_per_plane: bool
    channels_last: bool

    min_clips: int
    max_clips: int

    clips: list[vs.VideoNode]
    ref_clip: vs.VideoNode
    out_format: vs.VideoFormat

    fd: FD_T

    if TYPE_CHECKING:
        def process(self, f: vs.VideoFrame, src: Any, dst: Any, plane: int | None, n: int) -> None:
            raise NotImplementedError
    else:
        process: Callable[[PyPlugin[FD_T], vs.VideoFrame, Any, Any, int | None, int], None]

    def __class_getitem__(cls, fdata: Type[FD_T] | None = None) -> Type[PyPlugin[FD_T]]:
        class PyPluginInnerClass(cls):  # type: ignore
            filter_data = fdata

        return PyPluginInnerClass

    def __init__(
        self,
        ref_clip: vs.VideoNode,
        clips: list[vs.VideoNode] | None = None,
        *,
        filter_mode: FilterMode | None = None,
        options: PyPluginOptions | None = None,
        input_per_plane: bool | list[bool] | None = None,
        output_per_plane: bool | None = None,
        channels_last: bool | None = None,
        min_clips: int | None = None,
        max_clips: int | None = None,
        **kwargs: Any
    ) -> None:
        assert ref_clip.format

        arguments = [
            (filter_mode, 'filter_mode', FilterMode.Parallel),
            (options, 'options', PyPluginOptions()),
            (input_per_plane, 'input_per_plane', True),
            (output_per_plane, 'output_per_plane', True),
            (channels_last, 'channels_last', False),
            (min_clips, 'min_clips', 1),
            (max_clips, 'max_clips', -1)
        ]

        for value, name, default in arguments:
            if value is not None:
                setattr(self, name, value)
            elif not hasattr(self, name):
                setattr(self, name, default)

        self.out_format = ref_clip.format

        self.ref_clip = self.options.norm_clip(ref_clip)

        self.clips = [self.options.norm_clip(clip) for clip in clips] if clips else []

        self_annotations = self.__annotations__.keys()

        for name, value in list(kwargs.items()):
            if name in self_annotations:
                setattr(self, name, value)
                kwargs.pop(name)

        if self.filter_data and not isinstance(self.filter_data, TypeVar):
            self.fd = self.filter_data(**kwargs)  # type: ignore
        else:
            self.fd = None  # type: ignore

        n_clips = 1 + len(self.clips)

        class_name = self.__class__.__name__

        inputs_per_plane = self.input_per_plane

        if not isinstance(inputs_per_plane, list):
            inputs_per_plane = [inputs_per_plane]

        for _ in range((1 + len(self.clips)) - len(inputs_per_plane)):
            inputs_per_plane.append(inputs_per_plane[-1])

        self._input_per_plane = inputs_per_plane

        if ref_clip.format.num_planes == 1:
            self.output_per_plane = True

        self.is_single_plane = [
            bool(clip.format and clip.format.num_planes == 1)
            for clip in (self.ref_clip, *self.clips)
        ]

        if n_clips < self.min_clips or (self.max_clips > 0 and n_clips > self.max_clips):
            max_clips_str = 'inf' if self.max_clips == -1 else self.max_clips
            raise ValueError(
                f'{class_name}: You must pass {self.min_clips} <= n clips <= {max_clips_str}!'
            )

        if not self.output_per_plane and (ref_clip.format.subsampling_w or ref_clip.format.subsampling_h):
            raise ValueError(
                f'{class_name}: You can\'t have output_per_plane=False with a subsampled clip!'
            )

        for idx, clip, ipp in zip(count(-1), (self.ref_clip, *self.clips), self._input_per_plane):
            assert clip.format
            if not ipp and (clip.format.subsampling_w or clip.format.subsampling_h):
                clip_type = 'Ref Clip' if idx == -1 else f'Clip Index: {idx}'
                raise ValueError(
                    f'{class_name}: You can\'t have input_per_plane=False with a subsampled clip! ({clip_type})'
                )

    @PyPluginBackendBase.ensure_output
    def invoke(self) -> vs.VideoNode:
        assert self.ref_clip.format

        def _stack_frame(frame: vs.VideoFrame, idx: int) -> memoryview | list[memoryview]:
            return frame[0] if self.is_single_plane[idx] else [frame[p] for p in {0, 1, 2}]

        output_func: (
            Callable[[vs.VideoFrame, int], vs.VideoFrame] | Callable[[tuple[vs.VideoFrame, ...], int], vs.VideoFrame]
        )

        if self.output_per_plane:
            if self.clips:
                def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                    fout = f[0].copy()

                    pre_stacked_clips = {
                        idx: _stack_frame(frame, idx)
                        for idx, frame in enumerate(f)
                        if not self._input_per_plane[idx]
                    }

                    for p in range(fout.format.num_planes):
                        inputs_data = [
                            frame[p] if self._input_per_plane[idx] else pre_stacked_clips[idx]
                            for idx, frame in enumerate(f)
                        ]

                        self.process(fout, inputs_data, fout[p], p, n)

                    return fout
            else:
                if self._input_per_plane[0]:
                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        for p in range(fout.format.num_planes):
                            self.process(fout, f[p], fout[p], p, n)

                        return fout
                else:
                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        pre_stacked_clip = _stack_frame(f, 0)

                        for p in range(fout.format.num_planes):
                            self.process(fout, pre_stacked_clip, fout[p], p, n)

                        return fout
        else:
            if self.clips:
                def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                    fout = f[0].copy()

                    src_arrays = [_stack_frame(frame, idx) for idx, frame in enumerate(f)]

                    self.process(fout, src_arrays, fout, None, n)

                    return fout
            else:
                if self.ref_clip.format.num_planes == 1:
                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        self.process(fout, f[0], fout[0], 0, n)

                        return fout
                else:
                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        self.process(fout, f, fout, None, n)

                        return fout

        modify_frame_partial = partial(
            vs.core.std.ModifyFrame, self.ref_clip, (self.ref_clip, *self.clips), output_func
        )

        if self.filter_mode is FilterMode.Serial:
            output = modify_frame_partial()
        elif self.filter_mode is FilterMode.Parallel:
            output = self.ref_clip.std.FrameEval(lambda n: modify_frame_partial())
        else:
            if self.clips:
                output_func_multi = cast(Callable[[tuple[vs.VideoFrame, ...], int], vs.VideoFrame], output_func)

                @frame_eval_async(self.ref_clip)
                async def output(n: int) -> vs.VideoFrame:
                    return output_func_multi(await get_frames(self.ref_clip, *self.clips, frame_no=n), n)
            else:
                output_func_single = cast(Callable[[vs.VideoFrame, int], vs.VideoFrame], output_func)

                @frame_eval_async(self.ref_clip)
                async def output(n: int) -> vs.VideoFrame:
                    return output_func_single(await get_frame(self.ref_clip, n), n)

        return output

    def __call__(self, func: Callable[..., Any]) -> vs.VideoNode:
        this_args = {'self', 'f', 'src', 'dst', 'plane', 'n'}

        annotations = set(func.__annotations__.keys()) - {'return'}

        if not annotations:
            raise ValueError(f'{self.__class__.__name__}: You must type hint the function!')

        if annotations - this_args:
            raise ValueError(f'{self.__class__.__name__}: Unkown arguments specified!')

        miss_args = this_args - annotations

        if 'self' in annotations:
            func = partial(func, self)
            annotations.remove('self')

        if not miss_args:
            self.process = func  # type: ignore
        else:
            def _wrapper(f, src, dst, plane, n) -> Any:  # type: ignore
                curr_locals = locals()
                return func(**{name: curr_locals[name] for name in annotations})

            self.process = _wrapper  # type: ignore

        return self.invoke()


class PyPlugin(Generic[FD_T], PyPluginBase[FD_T, memoryview]):
    ...


class PyPluginUnavailableBackend(PyPlugin[FD_T]):
    @copy_signature(PyPlugin.__init__)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from .exceptions import UnavailableBackend

        raise UnavailableBackend(self.backend, self)
