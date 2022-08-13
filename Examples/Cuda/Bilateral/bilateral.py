from __future__ import annotations

from functools import lru_cache
from typing import Any

import vapoursynth as vs
from numpy.typing import NDArray
from stgfunc import set_output, source
from vspyplugin import PyPluginCuda
from vsutil import get_peak_value

core = vs.core


class BilateralFilter(PyPluginCuda[None]):
    cuda_kernel = './bilateral.cu', 'bilateral'

    input_per_plane = True
    output_per_plane = True
    float_processing = True

    kernel_size = 16
    use_shared_memory = True

    def process(self, src: NDArray[Any], dst: NDArray[Any], plane: int | None, n: int) -> None:
        self.kernel.bilateral[plane](src, dst)

    @lru_cache
    def calc_shared_mem(self, blk_size_w: int, blk_size_h: int, dtype_size: int) -> int:
        return (2 * self.bil_radius + blk_size_w) * (2 * self.bil_radius + blk_size_h) * dtype_size

    def __init__(self, clip: vs.VideoNode, sigmaS: float, sigmaR: float, radius: int | None) -> None:
        from math import e, log2

        assert clip.format

        sigmaS_scaled, sigmaR_scaled = [(-0.5 / (val * val)) * log2(e) for val in (sigmaS, sigmaR)]

        if radius is None:
            radius = max(1, round(sigmaS * 3))

        self.bil_radius = radius

        return super().__init__(
            clip, kernel_kwargs=dict(
                sigmaS=sigmaS_scaled, sigmaR=sigmaR_scaled,
                is_float=clip.format.sample_type is vs.FLOAT,
                peak=float(get_peak_value(clip)),
                radius=self.bil_radius
            )
        )


def bilateral(
    clip: vs.VideoNode, sigmaS: float = 3.0, sigmaR: float = 0.02, radius: int | None = None
) -> vs.VideoNode:
    return BilateralFilter(clip, sigmaS, sigmaR, radius).invoke()


# Test - Compare with the original c++ plugin this is based off
# From my benchmarks, it's 2x faster with real numbers, just 6% in the vacuum (BlankClip, with zeroes)

src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", 8, matrix_prop=1)
# src = src.std.ShufflePlanes(0, vs.GRAY)
src = src.resize.Bicubic(format=vs.YUV444P8)

# src = src.std.BlankClip(keep=True, length=100000)

set_output(src)
set_output(src.bilateralgpu_rtc.Bilateral())
set_output(bilateral(src))
