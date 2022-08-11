from __future__ import annotations

from typing import Any, Callable, cast

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

        n_cuda_streams: int = 1

        def to_device(self, f: vs.VideoFrame, idx: int, plane: int) -> None:
            if self.n_cuda_streams == 1:
                self.src_arrays[plane][idx].data.copy_from_host(  # type: ignore
                    f.get_read_ptr(plane), self.src_data_lengths[plane][idx]
                )
            else:
                for stream in self.cuda_streams:
                    self.src_arrays[plane][idx].data.copy_from_host_async(  # type: ignore
                        f.get_read_ptr(plane), self.src_data_lengths[plane][idx], stream
                    )

        def from_device(self, f: vs.VideoFrame, plane: int) -> None:
            if self.n_cuda_streams == 1:
                self.out_arrays[plane].data.copy_to_host(  # type: ignore
                    f.get_write_ptr(plane), self.out_data_lengths[plane]
                )
            else:
                for stream in self.cuda_streams:
                    self.out_arrays[plane].data.copy_to_host_async(  # type: ignore
                        f.get_write_ptr(plane), self.out_data_lengths[plane], stream
                    )

        def eval_single_clip_per_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fout = f.copy()

            for p in range(fout.format.num_planes):
                self.to_device(f, 0, p)
                self.process_single(self.src_arrays[p][0], self.out_arrays[p], n)
                self.from_device(fout, p)

            return fout

        def eval_single_clip_one_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fout = f.copy()

            self.to_device(f, 0, 0)
            self.process_single(self.src_arrays[0][0], self.out_arrays[0], n)
            self.from_device(fout, 0)

            return fout

        def eval_multi_clips_per_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fout = f[0].copy()
            f = f[self.omit_first_clip:]

            for p in range(fout.format.num_planes):
                for i, frame in enumerate(f):
                    self.to_device(frame, i, p)

                self.process_multi(self.src_arrays[p], self.out_arrays[p], n)
                self.from_device(fout, p)

            return fout

        def eval_multi_clips_one_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fout = f[0].copy()
            f = f[self.omit_first_clip:]

            for i, frame in enumerate(f):
                self.to_device(frame, i, 0)

            self.process_multi(self.src_arrays[0], self.out_arrays[0], n)
            self.from_device(fout, 0)

            return fout

        def _post_invoke(self, function: Callable[..., vs.VideoFrame]) -> vs.VideoNode:
            assert self.ref_clip.format

            def _alloc_arrays(clip: vs.VideoNode) -> list[NDArray[Any]]:
                assert clip.format

                return [
                    cp.zeros(
                        (plane.height, plane.width), self.get_dtype(clip), 'C'
                    ) for plane in (
                        [clip] if clip.format.num_planes == 1 else
                        cast(list[vs.VideoNode], clip.std.SplitPlanes())
                    )
                ]

            src_arrays = [_alloc_arrays(clip) for clip in self.clips]
            self.src_arrays = [
                [array[plane] for array in src_arrays]
                for plane in range(self.ref_clip.format.num_planes)
            ]
            self.src_data_lengths = [
                [
                    round((a.shape[0] * a.shape[1] * a.dtype.itemsize) / self.n_cuda_streams)
                    for a in arr
                ] for arr in self.src_arrays
            ]
            self.out_arrays = _alloc_arrays(self.ref_clip)
            self.out_data_lengths = [
                round((arr.shape[0] * arr.shape[1] * arr.dtype.itemsize) / self.n_cuda_streams)
                for arr in self.out_arrays
            ]

            return super()._post_invoke(function)

        def __init__(
            self, clips: vs.VideoNode | list[vs.VideoNode], ref_clip: vs.VideoNode | None = None, **kwargs: Any
        ) -> None:
            self.cuda_streams = [
                cp.cuda.Stream(null=False, ptds=False, non_blocking=True)
                for _ in range(self.n_cuda_streams)
            ]
            self.cuda_is_101 = 10010 <= cp.cuda.runtime.runtimeGetVersion()

            super().__init__(clips, ref_clip, **kwargs)

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginCupy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
