from __future__ import annotations

from typing import Any

import vapoursynth as vs

import numpy as np

from .base import FD_T, PyBackend, PyPlugin, PyPluginUnavailableBackend
from .helpers import frame_eval_async, get_frame, get_frames

__all__ = [
    'PyPluginNumpy'
]

this_backend = PyBackend.NUMPY

try:
    from numpy import dtype
    from numpy.core._multiarray_umath import array as nparray  # type: ignore
    from numpy.core._multiarray_umath import copyto as npcopyto
    from numpy.typing import NDArray

    class PyPluginNumpy(PyPlugin[FD_T]):
        backend = this_backend

        def to_host(self, f: vs.VideoFrame, plane: int, copy: bool = False) -> NDArray[Any]:
            return nparray(f[plane], copy=copy)  # type: ignore

        def from_host(self, src: Any, dst: vs.VideoFrame, copy: bool = False) -> None:
            for plane in range(dst.format.num_planes):
                npcopyto(
                    nparray(dst[plane], copy=copy),
                    src[:, :, plane] if self.channels_last else src[plane, :, :]
                )

        def get_dtype(self, clip: vs.VideoNode) -> dtype[Any]:
            assert clip.format

            stype = 'float' if clip.format.sample_type is vs.FLOAT else 'uint'
            bps = clip.format.bits_per_sample
            return dtype(f'{stype}{bps}')

        @PyPlugin.ensure_output
        def invoke(self) -> vs.VideoNode:
            assert self.ref_clip.format

            stack_axis = 2 if self.channels_last else 0

            is_single_plane = [
                bool(clip.format and clip.format.num_planes == 1)
                for clip in (self.ref_clip, *self.clips)
            ]

            def _stack_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                if is_single_plane[idx]:
                    return self.to_host(frame, 0)

                return np.stack(frame, axis=stack_axis)  # type: ignore

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

                            self.process(inputs_data, output_array, n)

                        return fout
                else:
                    if self._input_per_plane[0]:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            ref_frame = await get_frame(self.ref_clip, n)
                            fout = ref_frame.copy()

                            for p in range(fout.format.num_planes):
                                self.process(self.to_host(ref_frame, p), self.to_host(fout, p), n)

                            return fout
                    else:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            ref_frame = await get_frame(self.ref_clip, n)
                            fout = ref_frame.copy()

                            pre_stacked_clip = _stack_frame(ref_frame, 0)

                            for p in range(fout.format.num_planes):
                                self.process(pre_stacked_clip, self.to_host(fout, p), n)

                            return fout
            else:
                if self.clips:
                    @frame_eval_async(self.ref_clip)
                    async def output(n: int) -> vs.VideoFrame:
                        frames = await get_frames(self.ref_clip, *self.clips, frame_no=n)
                        fout = frames[0].copy()

                        src_arrays = [_stack_frame(frame, idx) for idx, frame in enumerate(frames)]

                        out_array = np.zeros_like(src_arrays[0])

                        self.process(src_arrays, out_array, n)

                        self.from_host(out_array, fout)

                        return fout
                else:
                    if self.ref_clip.format.num_planes == 1:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            self.process(self.to_host(frame, 0), self.to_host(fout, 0), n)

                            return fout
                    else:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            src = np.stack(frame, axis=stack_axis)  # type: ignore
                            dst = np.stack(fout, axis=stack_axis)  # type: ignore

                            self.process(src, dst, n)

                            self.from_host(dst, fout)

                            return fout

            return output

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginNumpy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
