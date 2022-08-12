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

        def process(self, src: NDArray[Any] | list[NDArray[Any]], dst: NDArray[Any], n: int) -> None:
            raise NotImplementedError

        def to_host(self, f: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
            return nparray(f[plane], copy=copy)

        def from_host(self, src: Any, dst: vs.VideoFrame, plane: int, copy: bool = False) -> Any:
            return npcopyto(nparray(dst[plane], copy=copy), src)

        def get_dtype(self, clip: vs.VideoNode) -> dtype[Any]:
            assert clip.format

            stype = 'float' if clip.format.sample_type is vs.FLOAT else 'uint'
            bps = clip.format.bits_per_sample
            return dtype(f'{stype}{bps}')

        @PyPlugin.ensure_output
        def invoke(self) -> vs.VideoNode:
            assert self.ref_clip.format

            n_clips = len(self.clips)

            function: Any

            if self.out_format.num_planes > 1 or self.out_format.subsampling_w or self.out_format.subsampling_h:
                if n_clips == 1:
                    def function(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        for p in range(fout.format.num_planes):
                            self.process(self.to_host(f, p), self.to_host(fout, p), n)

                        return fout
                else:
                    def function(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
                        fout = f[0].copy()
                        f = f[self.omit_first_clip:]

                        for p in range(fout.format.num_planes):
                            self.process([self.to_host(frame, p) for frame in f], self.to_host(fout, p), n)

                        return fout
            else:
                if n_clips == 1:
                    def function(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        self.process(self.to_host(f, 0), self.to_host(fout, 0), n)

                        return fout
                else:
                    def function(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
                        fout = f[0].copy()

                        f = f[self.omit_first_clip:]

                        self.process([self.to_host(frame, 0) for frame in f], self.to_host(fout, 0), n)

                        return fout

            return self.ref_clip.std.ModifyFrame(self.clips, function)

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginNumpy(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
