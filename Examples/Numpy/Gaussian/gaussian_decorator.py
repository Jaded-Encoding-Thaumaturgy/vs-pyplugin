from __future__ import annotations

from scipy.ndimage import gaussian_filter
from vssource import source
from vstools import set_output, vs

from vspyplugin import FilterMode, PyPluginNumpy


def gaussian(clip: vs.VideoNode, sigma: float | tuple[float, float] = 0.5) -> vs.VideoNode:
    if isinstance(sigma, tuple):
        sigma_h, sigma_v = sigma
    else:
        sigma_h = sigma_v = sigma

    @PyPluginNumpy(clip, filter_mode=FilterMode.Parallel)
    def output(src: PyPluginNumpy.DT, dst: PyPluginNumpy.DT) -> None:
        gaussian_filter(src, (sigma_v, sigma_h), output=dst)

    return output


src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", bits=8)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

set_output(src)
set_output(src.bilateral.Gaussian(1.5))
set_output(gaussian(src, 1.5))
