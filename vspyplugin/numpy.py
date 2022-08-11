from __future__ import annotations

from typing import Any

import vapoursynth as vs

from .base import FD_T, PyBackend, PyPlugin, PyPluginUnavailableBackend

__all__ = [
    'PyPluginNumpy'
]

this_backend = PyBackend.NUMPY

try:
    from numpy import dtype
    from numpy.core._multiarray_umath import array as nparray, copyto as npcopyto  # type: ignore
    from numpy.typing import NDArray

    class PyPluginNumpy(PyPlugin[FD_T]):
        backend = this_backend

        def process_single(self, src: NDArray[Any], dst: NDArray[Any], n: int) -> None:
            raise NotImplementedError

        def process_multi(self, srcs: list[NDArray[Any]], dst: NDArray[Any], n: int) -> None:
            raise NotImplementedError

        def to_host(self, f: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
            return nparray(f[plane], copy=copy)

        def from_host(self, src: Any, dst: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
            return npcopyto(nparray(dst[plane], copy=copy), src)

        def eval_single_clip_per_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fout = f.copy()

            for p in range(fout.format.num_planes):
                self.process_single(self.to_host(f, p), self.to_host(fout, p), n)

            return fout

        def eval_single_clip_one_plane(self, f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fout = f.copy()

            self.process_single(self.to_host(f, 0), self.to_host(fout, 0), n)

            return fout

        def eval_multi_clips_per_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fout = f[0].copy()
            f = f[self.omit_first_clip:]

            for p in range(fout.format.num_planes):
                self.process_multi([self.to_host(frame, p) for frame in f], self.to_host(fout, p), n)

            return fout

        def eval_multi_clips_one_plane(self, f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fout = f[0].copy()
            f = f[self.omit_first_clip:]

            self.process_multi([self.to_host(frame, 0) for frame in f], self.to_host(fout, 0), n)

            return fout

        def get_dtype(self, clip: vs.VideoNode) -> dtype[Any]:
            assert clip.format

            stype = 'float' if clip.format.sample_type is vs.FLOAT else 'uint'
            bps = clip.format.bits_per_sample
            return dtype(f'{stype}{bps}')

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginNumpy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
