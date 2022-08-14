from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar, cast

import vapoursynth as vs

from .backends import PyBackend
from .base import FD_T, PyPluginOptions, PyPluginUnavailableBackend
from .utils import get_c_dtype_long

__all__ = [
    'PyPluginCuda',
    'CudaCompileFlags', 'PyPluginCudaOptions'
]

this_backend = PyBackend.CUDA

T = TypeVar('T')


@dataclass
class CudaCompileFlags:
    std: Literal[3, 11, 14, 17, 20] = 17
    use_fast_math: bool = True
    extra_vectorization: bool = True
    options: tuple[str, ...] | None = None

    def to_tuple(self) -> tuple[str, ...]:
        options = [] if self.options is None else list(self.options)

        if self.use_fast_math:
            options.append('--use_fast_math')

        if self.std:
            options.append(f'--std=c++{self.std:02}')

        if self.extra_vectorization:
            options.append('--extra-device-vectorization')

        return tuple(set(options))


@dataclass
class PyPluginCudaOptions(PyPluginOptions):
    compile_flags: CudaCompileFlags = CudaCompileFlags()
    backend: Literal['nvrtc', 'nvcc'] = 'nvrtc'
    translate_cucomplex: bool = False
    enable_cooperative_groups: bool = False
    jitify: bool = False
    max_dynamic_shared_size_bytes: int | None = None
    preferred_shared_memory_carveout: int | None = None


try:
    from cupy import RawKernel
    from numpy.typing import NDArray

    from .cupy import PyPluginCupy

    class CudaKernelFunction:
        def __call__(
            self, src: NDArray[Any], dst: NDArray[Any], *args: Any,
            kernel_size: tuple[int, int] = ..., block_size: tuple[int, int] = ..., shared_mem: int = ...
        ) -> Any:
            ...

    class CudaKernelFunctionPlanes(CudaKernelFunction):
        __slots__ = ('function', 'planes_function')

        def __init__(
            self, function: CudaKernelFunction, planes_functions: list[CudaKernelFunction] | None = None
        ) -> None:
            self.function = function
            if planes_functions is None:
                self.planes_functions = [function]
            else:
                self.planes_functions = planes_functions

            self.planes_functions += self.planes_functions[-1:] * (3 - len(self.planes_functions))

        if not TYPE_CHECKING:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return self.function(*args, **kwargs)

        def __getitem__(self, plane: int | None) -> CudaKernelFunction:
            if plane is None:
                return self.function

            return self.planes_functions[plane]

    class CudaKernelFunctions:
        def __init__(self, **kwargs: CudaKernelFunctionPlanes) -> None:
            for key, func in kwargs.items():
                setattr(self, key, func)

        if TYPE_CHECKING:
            def __getattribute__(self, __name: str) -> CudaKernelFunctionPlanes:
                ...

    class PyPluginCuda(PyPluginCupy[FD_T]):
        backend = this_backend

        cuda_kernel: tuple[str | Path, str | Sequence[str]]

        kernel_size: int | tuple[int, int] = 16

        use_shared_memory: bool = False

        options: PyPluginCudaOptions = PyPluginCudaOptions()

        kernel_kwargs: dict[str, Any]

        kernel: CudaKernelFunctions

        @lru_cache
        def calc_shared_mem(self, plane: int, blk_size_w: int, blk_size_h: int, dtype_size: int) -> int:
            return blk_size_w * blk_size_h * dtype_size

        @lru_cache
        def normalize_kernel_size(
            self, plane: int, blk_size_w: int, blk_size_h: int, width: int, height: int
        ) -> tuple[int, int]:
            return ((width + blk_size_w - 1) // blk_size_w, (height + blk_size_h - 1) // blk_size_h)

        @lru_cache
        def get_kernel_size(self, plane: int, width: int, height: int) -> tuple[int, int]:
            if isinstance(self.kernel_size, tuple):
                block_x, block_y = self.kernel_size
            else:
                block_x = block_y = self.kernel_size

            return block_x, block_y

        def norm_kernel_args(self, value: Any) -> str:
            string = str(value)

            if isinstance(value, bool):
                return string.lower()

            return string

        def get_kernel_args(self, plane: int, width: int, height: int, **kwargs: Any) -> dict[str, Any]:
            from vsutil import get_peak_value, get_lowest_value, get_neutral_value

            assert self.ref_clip.format

            block_x, block_y = self.get_kernel_size(plane, width, height)

            kernel_args = dict[str, Any](
                use_shared_memory=self.use_shared_memory,
                block_x=block_x, block_y=block_y,
                data_type=get_c_dtype_long(self.ref_clip),
                is_float=self.ref_clip.format.sample_type is vs.FLOAT,
                lowest_value=float(get_lowest_value(self.ref_clip)),
                neutral_value=float(get_neutral_value(self.ref_clip)),
                peak_value=float(get_peak_value(self.ref_clip)),
            )

            if self.fd:
                try:
                    kernel_args |= self.fd  # type: ignore
                except BaseException:
                    ...

            return kwargs | kernel_args | dict(width=width, height=height)

        def __init__(
            self, ref_clip: vs.VideoNode, clips: list[vs.VideoNode] | None = None,
            kernel_kwargs: dict[str, Any] | None = None,
            kernel_planes_kwargs: list[dict[str, Any] | None] | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(ref_clip, clips, **kwargs)

            assert self.ref_clip.format

            if kernel_kwargs is None:
                kernel_kwargs = {}

            if kernel_planes_kwargs:
                kernel_planes_kwargs += kernel_planes_kwargs[-1:] * (3 - len(kernel_planes_kwargs))

            if not hasattr(self, 'cuda_kernel'):
                raise RuntimeError(f'{self.__class__.__name__}: You\'re missing cuda_kernel!')

            cuda_path, cuda_functions = self.cuda_kernel
            if isinstance(cuda_functions, str):
                cuda_functions = [cuda_functions]

            if not isinstance(cuda_path, Path):
                cuda_path = Path(cuda_path)

            cuda_path = cuda_path.absolute().resolve()

            cuda_kernel_code: str | None = None
            if cuda_path.exists():
                cuda_kernel_code = cuda_path.read_text()
            elif isinstance(self.cuda_kernel[0], str):
                cuda_kernel_code = self.cuda_kernel[0]

            if cuda_kernel_code:
                cuda_kernel_code = cuda_kernel_code.strip()

            if not cuda_kernel_code:
                raise RuntimeError(f'{self.__class__.__name__}: Cuda Kernel code not found!')

            def _wrap_kernel_function(
                def_kernel_size: tuple[int, int],
                def_block_size: tuple[int, int],
                def_shared_mem: int, function: Any
            ) -> CudaKernelFunction:
                def _inner_function(
                    *args: Any,
                    kernel_size: tuple[int, int] = def_kernel_size,
                    block_size: tuple[int, int] = def_block_size,
                    shared_mem: int = def_shared_mem
                ) -> Any:
                    return function(kernel_size, block_size, args, shared_mem=shared_mem)

                return cast(CudaKernelFunction, _inner_function)

            raw_kernel_kwargs = dict(
                options=('-Xptxas', '-O3', *self.options.compile_flags.to_tuple()),
                backend=self.options.backend,
                translate_cucomplex=self.options.translate_cucomplex,
                enable_cooperative_groups=self.options.enable_cooperative_groups,
                jitify=self.options.jitify
            )

            _cache_kernel_funcs = dict[tuple[int, str], CudaKernelFunction]()

            def _get_kernel_func(name: str, plane: int, width: int, height: int) -> CudaKernelFunction:
                assert self.ref_clip.format and cuda_kernel_code and kernel_kwargs

                kernel_args = self.get_kernel_args(plane, width, height, **kernel_kwargs)
                block_sizes: tuple[int, int] = kernel_args['block_x'], kernel_args['block_y']

                if kernel_planes_kwargs and (p_kwargs := kernel_planes_kwargs[plane]):
                    kernel_args |= p_kwargs

                kernel_args = {
                    name: self.norm_kernel_args(value)
                    for name, value in kernel_args.items()
                }

                def_kernel_size = self.normalize_kernel_size(
                    plane, *block_sizes, self.ref_clip.width, self.ref_clip.height
                )

                def_shared_mem = self.calc_shared_mem(
                    plane, *block_sizes, self.ref_clip.format.bytes_per_sample
                ) if self.use_shared_memory else 0

                sub_kernel_code = Template(cuda_kernel_code).substitute(kernel_args)

                kernel_key = hash(sub_kernel_code), name

                if kernel_key not in _cache_kernel_funcs:
                    kernel = RawKernel(code=sub_kernel_code, name=name, **raw_kernel_kwargs)

                    if self.options.max_dynamic_shared_size_bytes is not None:
                        kernel.max_dynamic_shared_size_bytes = self.options.max_dynamic_shared_size_bytes

                    if self.options.preferred_shared_memory_carveout is not None:
                        kernel.preferred_shared_memory_carveout = self.options.preferred_shared_memory_carveout

                    kernel.compile()

                    _cache_kernel_funcs[kernel_key] = _wrap_kernel_function(
                        def_kernel_size, block_sizes, def_shared_mem, kernel
                    )

                return _cache_kernel_funcs[kernel_key]

            chroma_res = (
                self.ref_clip.width // max(1, self.ref_clip.format.subsampling_w),
                self.ref_clip.height // max(1, self.ref_clip.format.subsampling_h)
            )

            resolutions = [(0, self.ref_clip.width, self.ref_clip.height), (1, *chroma_res), (2, *chroma_res)]

            kernel_functions = {
                name: [
                    _get_kernel_func(name, plane, width, height)
                    for plane, width, height in resolutions
                ] for name in cuda_functions
            }

            self.kernel = CudaKernelFunctions(**{
                name: CudaKernelFunctionPlanes(funcs[0], funcs)
                for name, funcs in kernel_functions.items()
            })

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginCuda(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
