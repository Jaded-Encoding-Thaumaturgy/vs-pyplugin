from __future__ import annotations

from dataclasses import dataclass

import vapoursynth as vs
from stgfunc import set_output, source
from vspyplugin import PyPluginCython
from vspyplugin.base import PyPluginOptions
from vspyplugin.types import FilterMode

core = vs.core


@dataclass
class SigmaFilterData:
    radius: int
    thr: float


class SigmaFilter(PyPluginCython[SigmaFilterData]):
    options = PyPluginOptions(float_processing=True)
    cython_kernel = 'sigma_filter'
    input_per_plane = True
    output_per_plane = True
    filter_mode = FilterMode.Async

    def process(self, f: vs.VideoFrame, src: memoryview, dst: memoryview, plane: int | None, n: int) -> None:
        self.kernel.sigma_filter(src, dst, self.fd.radius, self.fd.thr)


def sigma_filter(clip: vs.VideoNode, radius: int = 3, thr: float = 0.01) -> vs.VideoNode:
    return SigmaFilter(clip, radius=radius, thr=thr).invoke()


# Taken from
# https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/SigmaFilter_cython/sigma_filter.pyx

src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", 8, matrix_prop=1)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

set_output(src)
set_output(sigma_filter(src))
