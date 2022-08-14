from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import vapoursynth as vs
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from stgfunc import set_output, source
from vspyplugin import PyPluginNumpy

core = vs.core


@dataclass
class GaussianFilterData:
    sigma_h: float
    sigma_v: float


class GaussianFilter(PyPluginNumpy[GaussianFilterData]):
    def process(self, f: vs.VideoFrame, src: NDArray[Any], dst: NDArray[Any], plane: int | None, n: int) -> None:
        gaussian_filter(src, self.fd.sigma_v, output=dst)


def gaussian(clip: vs.VideoNode, sigma: float | tuple[float, float] = 0.5) -> vs.VideoNode:
    if isinstance(sigma, tuple):
        sigma_h, sigma_v = sigma
    else:
        sigma_h = sigma_v = sigma

    return GaussianFilter(clip, sigma_h=sigma_h, sigma_v=sigma_v).invoke()


# This is unoptimized as hell, but it's just for testing

src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", 8, matrix_prop=1)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

set_output(src)
set_output(src.bilateral.Gaussian(1.5))
set_output(gaussian(src, 1.5))
