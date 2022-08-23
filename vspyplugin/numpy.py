from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, cast

import vapoursynth as vs

from .abstracts import FD_T
from .backends import PyBackend
from .base import PyPlugin, PyPluginBase, PyPluginUnavailableBackend
from .types import OutputFunc_T, copy_signature
from .utils import get_resolutions

__all__ = [
    'PyPluginNumpyBase', 'PyPluginNumpy',
    'NDT_T'
]

this_backend = PyBackend.NUMPY

try:
    import numpy as np
    from numpy import dtype
    from numpy.core._multiarray_umath import copyto as npcopyto
    from numpy.typing import NDArray

    NDT_T = TypeVar('NDT_T', bound=NDArray[Any])

    if TYPE_CHECKING:
        concatenate: Callable[..., NDT_T]
    else:
        from numpy.core.numeric import concatenate

    _cache_dtypes = dict[int, dtype[Any]]()

    class PyPluginNumpyBase(PyPluginBase[FD_T, NDT_T]):
        backend = this_backend

        def to_host(self, f: vs.VideoFrame, plane: int) -> NDT_T:
            p = f[plane]
            return np.ndarray(p.shape, self.get_dtype(f), p)  # type: ignore

        def from_host(self, src: NDArray[Any], dst: vs.VideoFrame) -> None:
            for plane in range(dst.format.num_planes):
                npcopyto(self.to_host(dst, plane), src[self._slice_idxs[plane]])

        @staticmethod
        def get_dtype(clip: vs.VideoNode | vs.VideoFrame) -> dtype[Any]:
            fmt = cast(vs.VideoFormat, clip.format)

            if fmt.id not in _cache_dtypes:
                stype = 'float' if fmt.sample_type is vs.FLOAT else 'uint'
                _cache_dtypes[fmt.id] = dtype(f'{stype}{fmt.bits_per_sample}')

            return _cache_dtypes[fmt.id]

        @staticmethod
        def alloc_plane_arrays(clip: vs.VideoNode | vs.VideoFrame, fill: int | float | None = 0) -> list[NDT_T]:
            assert clip.format

            function = np.empty if fill is None else np.zeros if fill == 0 else partial(np.full, fill_value=fill)

            return [
                function((height, width), dtype=PyPluginNumpy.get_dtype(clip), order='C')  # type: ignore
                for _, width, height in get_resolutions(clip, True)
            ]

        def _get_data_len(self, arr: NDT_T) -> int:
            return arr.shape[0] * arr.shape[1] * arr.dtype.itemsize

        @copy_signature(PyPlugin.__init__)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

            no_slice = slice(None, None, None)
            self._slice_idxs = cast(list[slice], [
                (
                    [plane, no_slice][self.channels_last],
                    no_slice,
                    [no_slice, plane][self.channels_last]
                ) for plane in range(3)
            ])

        def _invoke_func(self) -> OutputFunc_T:
            assert self.ref_clip.format

            if self.channels_last:
                stack_slice = (slice(None), slice(None), None)

                def _stack_whole_frame(frame: vs.VideoFrame) -> NDT_T:
                    return concatenate([
                        self.to_host(frame, plane)[stack_slice] for plane in {0, 1, 2}
                    ], axis=2)
            else:
                def _stack_whole_frame(frame: vs.VideoFrame) -> NDT_T:
                    return concatenate([
                        self.to_host(frame, plane) for plane in {0, 1, 2}
                    ], axis=0)

            def _stack_frame(frame: vs.VideoFrame, idx: int) -> NDT_T:
                if self.is_single_plane[idx]:
                    return self.to_host(frame, 0)

                return _stack_whole_frame(frame)

            if not self.output_per_plane:
                dst_stacked_arr = np.zeros(
                    (self.ref_clip.height, self.ref_clip.width, 3), self.get_dtype(self.ref_clip)
                )

            output_func: OutputFunc_T

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
                            output_array = self.to_host(fout, p)

                            inputs_data = [
                                self.to_host(frame, p)
                                if self._input_per_plane[idx]
                                else pre_stacked_clips[idx]
                                for idx, frame in enumerate(f)
                            ]

                            self.process(fout, inputs_data, output_array, p, n)

                        return fout
                else:
                    if self._input_per_plane[0]:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            for p in range(fout.format.num_planes):
                                self.process(fout, self.to_host(f, p), self.to_host(fout, p), p, n)

                            return fout
                    else:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            pre_stacked_clip = _stack_frame(f, 0)

                            for p in range(fout.format.num_planes):
                                self.process(fout, pre_stacked_clip, self.to_host(fout, p), p, n)

                            return fout
            else:
                if self.clips:
                    def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                        fout = f[0].copy()

                        src_arrays = [_stack_frame(frame, idx) for idx, frame in enumerate(f)]

                        self.process(fout, src_arrays, dst_stacked_arr, None, n)

                        self.from_host(dst_stacked_arr, fout)

                        return fout
                else:
                    if self.ref_clip.format.num_planes == 1:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            self.process(fout, self.to_host(f, 0), self.to_host(fout, 0), 0, n)

                            return fout
                    else:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            self.process(fout, _stack_whole_frame(f), dst_stacked_arr, None, n)

                            self.from_host(dst_stacked_arr, fout)

                            return fout

            return output_func

    class PyPluginNumpy(Generic[FD_T], PyPluginNumpyBase[FD_T, NDArray[Any]]):
        ...

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginNumpy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
