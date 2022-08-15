from __future__ import annotations

from math import ceil
from typing import Any, TypeVar

import vapoursynth as vs
from numba import jit, prange  # type: ignore
from numpy.typing import NDArray
from stgfunc import set_output, source
from vspyplugin import PyPluginNumpy
from vsutil import scale_value

core = vs.core

F = TypeVar('F')


def erase_module(func: F) -> F:
    if hasattr(func, '__module__') and func.__module__ == '__vapoursynth__':
        func.__module__ = None  # type: ignore

    return func


@jit("int32(int32, int32)", nopython=True, nogil=True)  # type: ignore
@erase_module
def clamp(val: int, high: int) -> int:
    return min(max(val, 0), high)


@jit(
    nopython=True, nogil=True, fastmath=True, parallel=False,
    boundscheck=False, error_model='numpy', inline='always'
)  # type: ignore
@erase_module
def sigma_filter_numba(
    src: NDArray[Any], dst: NDArray[Any], radius: int, thr: float, height: int, width: int
) -> None:
    for y in prange(height):
        for x in prange(width):
            center = src[y, x]
            acc = 0.0
            count = 0

            for j in prange(-radius, radius + 1):
                for i in prange(-radius, radius + 1):
                    val = src[clamp(y + j, height - 1), clamp(x + i, width - 1)]

                    if abs(center - val) < thr:
                        acc += val
                        count += 1

            dst[y, x] = ceil(acc / count)


def sigma_filter(clip: vs.VideoNode, radius: int = 3, thr: float = 0.01) -> vs.VideoNode:
    assert clip.format

    height, width = clip.height, clip.width

    thr = scale_value(thr, 32, clip.format.bits_per_sample)

    @PyPluginNumpy(clip)
    def output(src: NDArray[Any], dst: NDArray[Any]) -> None:
        sigma_filter_numba(src, dst, radius, thr, height, width)

    return output


# Taken from
# https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/sigma_filter_numba.vpy

src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", 8, matrix_prop=1)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

set_output(src)
set_output(sigma_filter(src))
