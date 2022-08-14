from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import vapoursynth as vs

from .backends import PyBackend
from .base import FD_T, PyPlugin, PyPluginUnavailableBackend
from .coroutines import frame_eval_async, get_frame, get_frames, wait
from .utils import get_resolutions

__all__ = [
    'PyPluginCupy'
]

this_backend = PyBackend.CUPY


try:
    from cupy_backends.cuda.api import runtime  # type: ignore

    import cupy as cp
    from cupy import cuda
    from numpy.typing import NDArray

    from .numpy import PyPluginNumpy

    if TYPE_CHECKING:
        concatenate: Callable[..., NDArray[Any]]
    else:
        from cupy._core import concatenate_method as concatenate

    class PyPluginCupy(PyPluginNumpy[FD_T]):
        backend = this_backend

        cuda_num_streams: int = 0

        def _synchronize(self, events: list[cuda.Event]) -> None:
            for event in events:
                self.cuda_default_stream.wait_event(event)

            self.cuda_device.synchronize()

        def _memcpy_async(
            self, dst_ptr: int, src_ptr: int, amount: int, kind: int, sync: bool = True
        ) -> list[cuda.Event]:
            offset = 0

            stop_events = []
            for stream in self.cuda_streams:
                with stream:
                    runtime.memcpyAsync(dst_ptr + offset, src_ptr + offset, amount, kind, stream.ptr)

                    stop_events.append(cuda.Event())

                    offset += amount

            if sync:
                self._synchronize(stop_events)

            return stop_events

        def to_device(self, f: vs.VideoFrame, idx: int, plane: int) -> NDArray[Any]:
            self._memcpy_func(
                int(self.src_arrays[plane][idx].data),
                cast(int, f.get_read_ptr(plane).value),
                self.src_data_lengths[plane][idx],
                runtime.memcpyHostToDevice
            )

            return self.src_arrays[plane][idx]

        def from_device(self, dst: vs.VideoFrame) -> vs.VideoFrame:
            if self.cuda_num_streams:
                events = []
                for plane in range(dst.format.num_planes):
                    events.extend(
                        self._memcpy_async(
                            cast(int, dst.get_write_ptr(plane).value),
                            self._dst_pointers[plane],
                            self.out_data_lengths[plane],
                            runtime.memcpyDeviceToHost,
                            False
                        )
                    )
                self._synchronize(events)
            else:
                for plane in range(dst.format.num_planes):
                    runtime.memcpy(
                        cast(int, dst.get_write_ptr(plane).value),
                        self._dst_pointers[plane],
                        self.out_data_lengths[plane],
                        runtime.memcpyDeviceToHost
                    )

            return dst

        def _alloc_arrays(self, clip: vs.VideoNode) -> list[NDArray[Any]]:
            assert clip.format

            return [
                cp.zeros((height, width), self.get_dtype(clip), 'C')
                for _, width, height in get_resolutions(clip, True)
            ]

        def _get_data_len(self, arr: NDArray[Any]) -> int:
            return round(super()._get_data_len(arr) / max(1, self.cuda_num_streams))

        def __init__(
            self, ref_clip: vs.VideoNode, clips: list[vs.VideoNode] | None = None, **kwargs: Any
        ) -> None:
            super().__init__(ref_clip, clips, **kwargs)

            assert self.ref_clip.format

            if self.cuda_num_streams != 0 and self.cuda_num_streams < 2:
                raise ValueError(f'{self.__class__.__name__}: cuda_num_streams must be 0 or >= 2!')

            self.cuda_device = cuda.Device()
            self.cuda_memory_pool = cuda.MemoryPool()

            cuda.set_allocator(self.cuda_memory_pool.malloc)

            self.cuda_default_stream = cuda.Stream(non_blocking=True)
            self.cuda_streams = [cuda.Stream(non_blocking=True) for _ in range(self.cuda_num_streams)]

            self.cuda_is_101 = 10010 <= runtime.runtimeGetVersion()

            src_arrays = [self._alloc_arrays(clip) for clip in (self.ref_clip, *self.clips)]
            self.src_arrays = [
                [array[plane] for array in src_arrays] for plane in range(self.ref_clip.format.num_planes)
            ]
            self.src_data_lengths = [[self._get_data_len(a) for a in arr] for arr in self.src_arrays]

            self.out_arrays = self._alloc_arrays(self.ref_clip)
            self.out_data_lengths = [self._get_data_len(arr) for arr in self.out_arrays]

        @PyPlugin.ensure_output
        def invoke(self) -> vs.VideoNode:
            assert self.ref_clip.format

            self._memcpy_func = self._memcpy_async if self.cuda_num_streams else runtime.memcpy

            if self.ref_clip.format.num_planes == 1:
                def _stack_whole_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                    return self.to_device(frame, idx, 0)
            elif self.channels_last:
                stack_slice = (slice(None), slice(None), None)

                def _stack_whole_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                    return concatenate([
                        self.to_device(frame, idx, plane)[stack_slice] for plane in {0, 1, 2}
                    ], axis=2)
            else:
                def _stack_whole_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                    return concatenate([
                        self.to_device(frame, idx, plane) for plane in {0, 1, 2}
                    ], axis=0)

            def _stack_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                if self.is_single_plane[idx]:
                    return self.to_device(frame, idx, 0)

                return _stack_whole_frame(frame, idx)

            if self.output_per_plane:
                dst_stacked_planes = self._alloc_arrays(self.ref_clip)
            else:
                shape = (self.ref_clip.height, self.ref_clip.width)

                shape_channels: tuple[int, ...]
                if self.is_single_plane[0]:
                    shape_channels = shape + (1, )
                elif self.channels_last:
                    shape_channels = shape + (3, )
                else:
                    shape_channels = (3, ) + shape

                dst_stacked_arr = cp.zeros(shape_channels, self.get_dtype(self.ref_clip))
                dst_stacked_planes = [
                    dst_stacked_arr[self._slice_idxs[plane]]
                    for plane in range(self.ref_clip.format.num_planes)
                ]

            self._dst_pointers = [int(source.data) for source in dst_stacked_planes]

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
                            inputs_data = [
                                self.to_device(frame, idx, p)
                                if self._input_per_plane[idx]
                                else pre_stacked_clips[idx]
                                for idx, frame in enumerate(frames)
                            ]

                            self.process(inputs_data, dst_stacked_planes[p], p, n)

                        return self.from_device(fout)
                else:
                    if self._input_per_plane[0]:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            for p in range(fout.format.num_planes):
                                self.process(self.to_device(frame, 0, p), dst_stacked_planes[p], p, n)

                            return self.from_device(fout)
                    else:
                        @frame_eval_async(self.ref_clip)
                        async def output(n: int) -> vs.VideoFrame:
                            frame = await get_frame(self.ref_clip, n)
                            fout = frame.copy()

                            pre_stacked_clip = _stack_frame(frame, 0)

                            for p in range(fout.format.num_planes):
                                self.process(pre_stacked_clip, dst_stacked_planes[p], p, n)

                            return self.from_device(fout)
            else:
                if self.clips:
                    async def inner_stack(clip: vs.VideoNode, n: int, idx: int) -> NDArray[Any]:
                        return _stack_frame(await get_frame(clip, n), idx)

                    @frame_eval_async(self.ref_clip)
                    async def output(n: int) -> vs.VideoFrame:
                        frame = await get_frame(self.ref_clip, n)
                        fout = frame.copy()

                        self.process(await wait(
                            inner_stack(clip, n, idx) for idx, clip in enumerate(self.clips, 1)
                        ), dst_stacked_arr, None, n)

                        return self.from_device(fout)
                else:
                    @frame_eval_async(self.ref_clip)
                    async def output(n: int) -> vs.VideoFrame:
                        frame = await get_frame(self.ref_clip, n)
                        fout = frame.copy()

                        self.process(_stack_whole_frame(frame, 0), dst_stacked_arr, None, n)

                        return self.from_device(fout)

            return output

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginCupy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
