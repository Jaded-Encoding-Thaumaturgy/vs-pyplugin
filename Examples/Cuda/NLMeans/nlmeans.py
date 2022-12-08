from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Generator

from stgfunc import source
from vstools import set_output, vs

from vspyplugin import FilterMode, ProcessMode, PyPluginCuda, PyPluginCudaOptions


@dataclass
class KNLMeansFilterData:
    ocl_x: int
    ocl_y: int
    coordinates: list[tuple[int, int]]

    def get_arrays(self, f: vs.VideoFrame) -> Generator[list[PyPluginCuda.DT], None, None]:
        yield PyPluginCuda.alloc_plane_arrays(f, 0)
        yield PyPluginCuda.alloc_plane_arrays(f, 0)
        yield PyPluginCuda.alloc_plane_arrays(f, None)
        yield PyPluginCuda.alloc_plane_arrays(f, None)
        yield PyPluginCuda.alloc_plane_arrays(f, 1.1920928955078125e-7)


class KNLMeansFilter(PyPluginCuda[KNLMeansFilterData]):
    cuda_kernel = 'nlmeans.cu', (
        'nlmDistance', 'nlmHorizontal', 'nlmVertical', 'nlmAccumulation', 'nlmFinish',
    )
    filter_mode = FilterMode.Serial
    input_per_plane = output_per_plane = channels_last = False
    options = PyPluginCudaOptions(force_precision=32, shift_chroma=True)

    @lru_cache
    def get_kernel_size(self, plane: int, func_name: str, width: int, height: int) -> tuple[int, ...]:
        if func_name in {'nlmHorizontal', 'nlmVertical'}:
            return (self.fd.ocl_x, self.fd.ocl_y, 1)

        return (32, 32, 1)

    @lru_cache
    def normalize_kernel_size(
        self, plane: int, func_name: str, blk_size_w: int, blk_size_h: int, width: int, height: int
    ) -> tuple[int, ...]:
        return ((width + blk_size_w - 1) // blk_size_w, (height + blk_size_h - 1) // blk_size_h)

    @PyPluginCuda.process(ProcessMode.SingleSrcIPF)  # type: ignore
    def _(self, src: KNLMeansFilter.DTL, dst: KNLMeansFilter.DTL, f: vs.VideoFrame, n: int) -> None:
        U2a, U2b, U4a, U4b, U5 = self.fd.get_arrays(f)

        for plane in range(f.format.num_planes):
            for j, i in self.fd.coordinates:
                self.kernel.nlmDistance[plane](src[plane], U4a[plane], i, j)
                self.kernel.nlmHorizontal[plane](U4a[plane], U4b[plane])
                self.kernel.nlmVertical[plane](U4b[plane], U4a[plane])
                self.kernel.nlmAccumulation[plane](
                    src[plane], U2a[plane], U2b[plane], U4a[plane], U5[plane], i, j
                )

            self.kernel.nlmFinish[plane](src[plane], dst[plane], U2a[plane], U2b[plane], U5[plane])


def knlmeans(
    clip: vs.VideoNode,
    a: int = 1, s: int = 4, h: float = 1.2,
    wmode: int = 0, wref: float = 1.0,
    ocl_x: int = 16, ocl_y: int = 8, ocl_r: int = 3
) -> vs.VideoNode:
    coordinates = [(j, i) for i in range(-a, a + 1) for j in range(-a, a + 1) if (j * (2 * a + 1) + i < 0)]

    return KNLMeansFilter(
        clip, coordinates=coordinates, ocl_x=ocl_x, ocl_y=ocl_y,
        use_shared_memory=True, kernel_kwargs=dict(
            s=s, h=h, wmode=wmode, wref=wref,
            hrz_block_x=ocl_x, hrz_block_y=ocl_y, hrz_result=ocl_r,
            vrt_block_x=ocl_x, vrt_block_y=ocl_y, vrt_result=ocl_r
        )
    ).invoke()

# https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/KNLMeasCL_cupy/knlm.cu


src = source(r"E:\Desktop\Encoding Sources\[BDMV] Takagi-San 3\TAKAGISAN3_1\BDMV\STREAM\00003.m2ts", 8, matrix_prop=1)
src = src.std.ShufflePlanes(0, vs.GRAY)
# src = src.resize.Bicubic(format=vs.YUV444P8)

# This doesn't work with chroma


set_output(src)
set_output(src.knlm.KNLMeansCL(d=0, a=4, s=4, h=1.2, wmode=0, ocl_x=16, ocl_y=8, ocl_r=3))
set_output(knlmeans(src, a=4, s=4, h=1.2, wmode=0, ocl_x=16, ocl_y=8, ocl_r=3))
