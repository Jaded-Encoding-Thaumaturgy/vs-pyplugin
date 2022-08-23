from __future__ import annotations

from functools import lru_cache
from math import e, log2

import vapoursynth as vs
from stgfunc import set_output, source
from vspyplugin import PyPluginCuda

core = vs.core


class BilateralFilter(PyPluginCuda[None]):
    radius: int

    @lru_cache
    def get_kernel_shared_mem(
        self, plane: int, func_name: str, blk_size_w: int, blk_size_h: int, dtype_size: int
    ) -> int:
        return (2 * self.radius + blk_size_w) * (2 * self.radius + blk_size_h) * dtype_size


def bilateral(
    clip: vs.VideoNode, sigmaS: float = 3.0, sigmaR: float = 0.02, radius: int | None = None
) -> vs.VideoNode:
    sigmaS_scaled, sigmaR_scaled = [(-0.5 / (val * val)) * log2(e) for val in (sigmaS, sigmaR)]

    if radius is None:
        radius = max(1, round(sigmaS * 3))

    kernel_kwargs = dict(sigmaS=sigmaS_scaled, sigmaR=sigmaR_scaled, radius=radius)

    @BilateralFilter(clip, None, 'bilateral', 16, True, radius=radius, kernel_kwargs=kernel_kwargs)
    def output(self: BilateralFilter, src: BilateralFilter.DT, dst: BilateralFilter.DT, plane: int) -> None:
        self.kernel.bilateral[plane](src, dst)

    return output

# Test - Compare with the original c++ plugin this is based off
# From my benchmarks, it's 2x faster with real numbers, just 6% in the vacuum (BlankClip, with zeroes)


src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", 8, matrix_prop=1)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

# src = src.std.BlankClip(keep=True, length=100000)

set_output(src)
set_output(src.bilateralgpu_rtc.Bilateral())
set_output(bilateral(src))
