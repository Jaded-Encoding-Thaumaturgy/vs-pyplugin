from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import vapoursynth as vs

import numpy as np

from .backends import PyBackend
from .base import FD_T, PyPlugin, PyPluginUnavailableBackend
from .coroutines import frame_eval_async, get_frame, get_frames

__all__ = [
    'PyPluginNumpy'
]

this_backend = PyBackend.NUMPY

try:
    from numpy import dtype
    from numpy.core._multiarray_umath import copyto as npcopyto
    from numpy.typing import NDArray

    if TYPE_CHECKING:
        concatenate: Callable[..., NDArray[Any]]
    else:
        from numpy.core.numeric import concatenate

    class PyPluginNumpy(PyPlugin[FD_T]):
        backend = this_backend

        def to_host(self, f: vs.VideoFrame, plane: int) -> NDArray[Any]:
            p = f[plane]
            return np.ndarray(p.shape, self.get_dtype(f), p)  # type: ignore

        def from_host(self, src: NDArray[Any], dst: vs.VideoFrame) -> None:
            for plane in range(dst.format.num_planes):
                npcopyto(self.to_host(dst, plane), src[self._slice_idxs[plane]])

        _cache_dtypes = dict[int, dtype[Any]]()

        def get_dtype(self, clip: vs.VideoNode | vs.VideoFrame) -> dtype[Any]:
            fmt = cast(vs.VideoFormat, clip.format)

            if fmt.id not in self._cache_dtypes:
                stype = 'float' if fmt.sample_type is vs.FLOAT else 'uint'
                self._cache_dtypes[fmt.id] = dtype(f'{stype}{fmt.bits_per_sample}')

            return self._cache_dtypes[fmt.id]

        def _get_data_len(self, arr: NDArray[Any]) -> int:
            return arr.shape[0] * arr.shape[1] * arr.dtype.itemsize

        def __init__(self, ref_clip: vs.VideoNode, clips: list[vs.VideoNode] | None = None, **kwargs: Any) -> None:
            super().__init__(ref_clip, clips, **kwargs)

            no_slice = slice(None, None, None)
            self._slice_idxs = cast(list[slice], [
                (
                    [plane, no_slice][self.channels_last],
                    no_slice,
                    [no_slice, plane][self.channels_last]
                ) for plane in range(3)
            ])

        @PyPlugin.ensure_output
        def invoke(self) -> vs.VideoNode:
            assert self.ref_clip.format

            if self.channels_last:
                stack_slice = (slice(None), slice(None), None)

                def _stack_whole_frame(frame: vs.VideoFrame) -> NDArray[Any]:
                    return concatenate([
                        self.to_host(frame, plane)[stack_slice] for plane in {0, 1, 2}
                    ], axis=2)
            else:
                def _stack_whole_frame(frame: vs.VideoFrame) -> NDArray[Any]:
                    return concatenate([
                        self.to_host(frame, plane) for plane in {0, 1, 2}
                    ], axis=0)

            def _stack_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                if self.is_single_plane[idx]:
                    return self.to_host(frame, 0)

                return _stack_whole_frame(frame)

            if self.output_per_plane:
                if self.clips:
                    @frame_eval_async(self.ref_clip)
                    async def output(n: int) -> vs.VideoFrame:
                        frames = await get_frames(self.ref_clip, *self.clips, frame_no=n)
                        fout = frames[0].copy()

                        pre_stacked_clips = {
                            idx: _stack_frame(frame, idx)
                            for idx, frame in enumerate(frames)
                            if not self._input_per_plane[idx]
                        }

                        for p in range(fout.format.num_planes):
                            output_array = self.to_host(fout, p)

                            inputs_data = [
                                self.to_host(frame, p)
                                if self._input_per_plane[idx]
                                else pre_stacked_clips[idx]
                                for idx, frame in enumerate(frames)
                            ]

                            self.process(fout, inputs_data, output_array, p, n)

                        return fout
                else:
                    if self._input_per_plane[0]:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            for p in range(fout.format.num_planes):
                                self.process(fout, self.to_host(frame, p), self.to_host(fout, p), p, n)

                            return fout
                    else:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            pre_stacked_clip = _stack_frame(frame, 0)

                            for p in range(fout.format.num_planes):
                                self.process(fout, pre_stacked_clip, self.to_host(fout, p), p, n)

                            return fout
            else:
                dst_stacked_arr = np.zeros(
                    (self.ref_clip.height, self.ref_clip.width, 3), self.get_dtype(self.ref_clip)
                )

                if self.clips:
                    @frame_eval_async(self.ref_clip)
                    async def output(n: int) -> vs.VideoFrame:
                        frames = await get_frames(self.ref_clip, *self.clips, frame_no=n)
                        fout = frames[0].copy()

                        src_arrays = [_stack_frame(frame, idx) for idx, frame in enumerate(frames)]

                        self.process(fout, src_arrays, dst_stacked_arr, None, n)

                        self.from_host(dst_stacked_arr, fout)

                        return fout
                else:
                    if self.ref_clip.format.num_planes == 1:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            self.process(fout, self.to_host(frame, 0), self.to_host(fout, 0), 0, n)

                            return fout
                    else:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            self.process(fout, _stack_whole_frame(frame), dst_stacked_arr, None, n)

                            self.from_host(dst_stacked_arr, fout)

                            return fout

            return output

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginNumpy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
