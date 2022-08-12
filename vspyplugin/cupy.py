from __future__ import annotations

from typing import Any, cast

import vapoursynth as vs

from .base import FD_T, PyBackend, PyPluginUnavailableBackend

__all__ = [
    'PyPluginCupy'
]

this_backend = PyBackend.CUPY


try:
    import cupy as cp
    from numpy.typing import NDArray

    from .numpy import PyPluginNumpy

    class PyPluginCupy(PyPluginNumpy[FD_T]):
        backend = this_backend

        cuda_streams_num: int = 0

        def to_device(self, f: vs.VideoFrame, idx: int, plane: int) -> None:
            if self.cuda_streams_num:
                stop_events = []
                for stream in self.cuda_streams:
                    self.src_arrays[plane][idx].data.copy_from_host_async(  # type: ignore
                        f.get_read_ptr(plane), self.src_data_lengths[plane][idx], stream
                    )

                    stop_events.append(stream.record())

                for stop_event in stop_events:
                    self.cuda_default_stream.wait_event(stop_event)

                self.cuda_device.synchronize()
            else:
                self.src_arrays[plane][idx].data.copy_from_host(  # type: ignore
                    f.get_read_ptr(plane), self.src_data_lengths[plane][idx]
                )

        def from_device(self, f: vs.VideoFrame, plane: int) -> None:
            if self.cuda_streams_num:
                stop_events = []
                for stream in self.cuda_streams:
                    self.out_arrays[plane].data.copy_to_host_async(  # type: ignore
                        f.get_write_ptr(plane), self.out_data_lengths[plane], stream
                    )

                    stop_events.append(stream.record())

                for stop_event in stop_events:
                    self.cuda_default_stream.wait_event(stop_event)

                self.cuda_device.synchronize()
            else:
                self.out_arrays[plane].data.copy_to_host(  # type: ignore
                    f.get_write_ptr(plane), self.out_data_lengths[plane]
                )

        def eval_single_clip_per_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fout = f.copy()

            for p in range(fout.format.num_planes):
                self.to_device(f, 0, p)
                self.process(self.src_arrays[p][0], self.out_arrays[p], n)
                self.from_device(fout, p)

            return fout

        def eval_single_clip_one_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fout = f.copy()

            self.to_device(f, 0, 0)
            self.process(self.src_arrays[0][0], self.out_arrays[0], n)
            self.from_device(fout, 0)

            return fout

        def eval_multi_clips_per_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fout = f[0].copy()
            f = f[self.omit_first_clip:]

            for p in range(fout.format.num_planes):
                for i, frame in enumerate(f):
                    self.to_device(frame, i, p)

                self.process(self.src_arrays[p], self.out_arrays[p], n)
                self.from_device(fout, p)

            return fout

        def eval_multi_clips_one_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fout = f[0].copy()
            f = f[self.omit_first_clip:]

            for i, frame in enumerate(f):
                self.to_device(frame, i, 0)

            self.process(self.src_arrays[0], self.out_arrays[0], n)
            self.from_device(fout, 0)

            return fout

        def _alloc_arrays(self, clip: vs.VideoNode) -> list[NDArray[Any]]:
            assert clip.format

            return [
                cp.zeros(
                    (plane.height, plane.width), self.get_dtype(clip), 'C'
                ) for plane in (
                    [clip] if clip.format.num_planes == 1 else
                    cast(list[vs.VideoNode], clip.std.SplitPlanes())
                )
            ]

        def _get_data_len(self, arr: NDArray[Any]) -> int:
            return round(
                (arr.shape[0] * arr.shape[1] * arr.dtype.itemsize)
                / max(1, self.cuda_streams_num)
            )

        def __init__(
            self, clips: vs.VideoNode | list[vs.VideoNode], ref_clip: vs.VideoNode | None = None, **kwargs: Any
        ) -> None:
            super().__init__(clips, ref_clip, **kwargs)
            assert self.ref_clip.format

            if self.cuda_streams_num != 0 and self.cuda_streams_num < 2:
                raise ValueError(f'{self.__class__.__name__}: cuda_streams_num must be 0 or >= 2!')

            self.cuda_device = cp.cuda.Device()
            self.cuda_memory_pool = cp.cuda.MemoryPool()

            cp.cuda.set_allocator(self.cuda_memory_pool.malloc)

            self.cuda_default_stream = cp.cuda.Stream(non_blocking=True)
            self.cuda_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self.cuda_streams_num)]
            self.cuda_is_101 = 10010 <= cp.cuda.runtime.runtimeGetVersion()

            src_arrays = [self._alloc_arrays(clip) for clip in self.clips]
            self.src_arrays = [
                [array[plane] for array in src_arrays] for plane in range(self.ref_clip.format.num_planes)
            ]
            self.out_arrays = self._alloc_arrays(self.ref_clip)

            self.src_data_lengths = [[self._get_data_len(a) for a in arr] for arr in self.src_arrays]
            self.out_data_lengths = [self._get_data_len(arr) for arr in self.out_arrays]

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginCupy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
