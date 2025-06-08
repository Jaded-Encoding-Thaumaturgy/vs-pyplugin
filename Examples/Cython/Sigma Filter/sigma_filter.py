from __future__ import annotations

from dataclasses import dataclass

from vssource import source
from vstools import set_output, vs

from vspyplugin import FilterMode, ProcessMode, PyPluginCython, PyPluginOptions


@dataclass
class SigmaFilterData:
    radius: int
    thr: float


class SigmaFilter(PyPluginCython[SigmaFilterData]):
    options = PyPluginOptions(force_precision=32)
    cython_kernel = 'sigma_filter'
    input_per_plane = True
    output_per_plane = True
    filter_mode = FilterMode.Async

    @PyPluginCython.process(ProcessMode.SingleSrcIPP)
    def _(self, src: SigmaFilter.DT, dst: SigmaFilter.DT, f: vs.VideoFrame, plane: int, n: int) -> None:
        self.kernel.sigma_filter(src, dst, self.fd.radius, self.fd.thr)


def sigma_filter(clip: vs.VideoNode, radius: int = 3, thr: float = 0.01) -> vs.VideoNode:
    return SigmaFilter(clip, radius=radius, thr=thr).invoke()


# Taken from
# https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/SigmaFilter_cython/sigma_filter.pyx

src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", bits=8)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

set_output(src)
set_output(sigma_filter(src))
