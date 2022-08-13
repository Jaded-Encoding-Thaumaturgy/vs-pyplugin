from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Literal, NamedTuple, Type, TypeVar, cast

import vapoursynth as vs

from .base import FD_T, GenericFilterData, PyBackend, PyPluginUnavailableBackend
from .utils import get_c_dtype_long

__all__ = [
    'PyPluginCuda',
    'CudaFunctions',
    'CudaCompileFlags', 'CudaOptions'
]

this_backend = PyBackend.CUDA


CF_T = TypeVar('CF_T')

CudaFunctions = NamedTuple


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
class CudaOptions:
    backend: Literal['nvrtc', 'nvcc'] = 'nvrtc'
    translate_cucomplex: bool = False
    enable_cooperative_groups: bool = False
    jitify: bool = False
    max_dynamic_shared_size_bytes: int | None = None
    preferred_shared_memory_carveout: int | None = None


try:
    from cupy import RawKernel

    from .cupy import PyPluginCupy

    class PyPluginCuda(PyPluginCupy[FD_T, CF_T]):  # type: ignore
        backend = this_backend

        cuda_kernel: str | Path

        kernel_size: int | tuple[int, int] = 16

        use_shared_memory: bool = False

        cuda_options: CudaOptions = CudaOptions()
        cuda_flags: CudaCompileFlags = CudaCompileFlags()

        kernel_kwargs: dict[str, Any]

        kernel_type: Type[CF_T]
        kernel: CF_T

        def __class_getitem__(  # type: ignore
            cls, fdata: tuple[Type[FD_T], Type[CF_T]] | None = None
        ) -> Type[PyPluginCuda[FD_T, CF_T]]:
            if fdata is None:
                raise RuntimeError(f'{cls.__class__.__name__}: You must specify the Cuda kernel functions!')

            if isinstance(fdata, tuple):
                filter_dtype = cast(Any, fdata[0])
                kernel_dtype = fdata[1]
            else:
                filter_dtype = GenericFilterData  # type: ignore
                kernel_dtype = fdata[0]

            class PyPluginCudaInnerClass(cls):  # type: ignore
                filter_data = filter_dtype
                kernel_type = kernel_dtype

            return PyPluginCudaInnerClass

        @lru_cache
        def calc_shared_mem(self, blk_size_w: int, blk_size_h: int, dtype_size: int) -> int:
            return blk_size_w * blk_size_h * dtype_size

        @lru_cache
        def normalize_kernel_size(
            self, blk_size_w: int, blk_size_h: int, width: int, height: int
        ) -> tuple[int, int]:
            return ((width + blk_size_w - 1) // blk_size_w, (height + blk_size_h - 1) // blk_size_h)

        def get_kernel_size(self) -> tuple[int, int]:
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

        def __init__(
            self, ref_clip: vs.VideoNode, clips: list[vs.VideoNode] | None = None,
            kernel_kwargs: dict[str, Any] | None = None, **kwargs: Any
        ) -> None:
            super().__init__(ref_clip, clips, **kwargs)
            assert self.ref_clip.format

            if kernel_kwargs is None:
                kernel_kwargs = {}

            cuda_kernel_code: str | None = None

            if not hasattr(self, 'cuda_kernel'):
                raise RuntimeError(f'{self.__class__.__name__}: You\'re missing cuda_kernel!')

            if isinstance(self.cuda_kernel, Path):
                cuda_path = self.cuda_kernel
            else:
                cuda_path = Path(self.cuda_kernel)

            cuda_path = cuda_path.absolute().resolve()

            if cuda_path.exists():
                cuda_kernel_code = cuda_path.read_text()
            elif isinstance(self.cuda_kernel, str):
                cuda_kernel_code = self.cuda_kernel

            if cuda_kernel_code:
                cuda_kernel_code = cuda_kernel_code.strip()

            if not cuda_kernel_code:
                raise RuntimeError(f'{self.__class__.__name__}: Cuda Kernel code not found!')

            self.cuda_kernel_code = cuda_kernel_code
            self.kernel_kwargs = kernel_kwargs

            block_x, block_y = self.get_kernel_size()

            kernel_args = dict(
                width=self.ref_clip.width, height=self.ref_clip.height,
                use_shared_memory=self.use_shared_memory,
                block_x=block_x, block_y=block_y,
                data_type=get_c_dtype_long(self.ref_clip)
            )

            try:
                kernel_args |= self.fd  # type: ignore
            except BaseException:
                ...

            if self.kernel_kwargs:
                kernel_args |= self.kernel_kwargs

            kernel_args = {
                name: self.norm_kernel_args(value)
                for name, value in kernel_args.items()
            }

            cuda_kernel_code = Template(self.cuda_kernel_code).substitute(kernel_args)

            default_options = (
                '-Xptxas', '-O3',
            )

            raw_kernel_kwargs = dict(
                options=(*default_options, *self.cuda_flags.to_tuple()),
                backend=self.cuda_options.backend,
                translate_cucomplex=self.cuda_options.translate_cucomplex,
                enable_cooperative_groups=self.cuda_options.enable_cooperative_groups,
                jitify=self.cuda_options.jitify
            )

            self.kernel_functions = {
                name: RawKernel(code=cuda_kernel_code, name=name, **raw_kernel_kwargs)
                for name in self.kernel_type.__annotations__.keys()
            }

            for kernel in self.kernel_functions.values():
                if self.cuda_options.max_dynamic_shared_size_bytes is not None:
                    kernel.max_dynamic_shared_size_bytes = self.cuda_options.max_dynamic_shared_size_bytes

                if self.cuda_options.preferred_shared_memory_carveout is not None:
                    kernel.preferred_shared_memory_carveout = self.cuda_options.preferred_shared_memory_carveout

                kernel.compile()

            def _wrap_kernel_function(
                def_kernel_size: tuple[int, int],
                def_block_size: tuple[int, int],
                def_shared_mem: int, function: Any
            ) -> Any:
                def _inner_function(
                    *args: Any,
                    kernel_size: tuple[int, int] = def_kernel_size,
                    block_size: tuple[int, int] = def_block_size,
                    shared_mem: int = def_shared_mem
                ) -> Any:
                    return function(kernel_size, block_size, args, shared_mem=shared_mem)

                return _inner_function

            block_x, block_y = self.get_kernel_size()

            def_kernel_size = self.normalize_kernel_size(
                block_x, block_y, self.ref_clip.width, self.ref_clip.height
            )

            def_shared_mem = self.calc_shared_mem(
                block_x, block_y, self.ref_clip.format.bytes_per_sample
            ) if self.use_shared_memory else 0

            self.kernel = self.kernel_type(**{
                name: _wrap_kernel_function(
                    def_kernel_size, (block_x, block_y), def_shared_mem, function
                ) for name, function in self.kernel_functions.items()
            })

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginCuda(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
