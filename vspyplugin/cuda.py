from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

import vapoursynth as vs

from .base import FD_T, PyBackend, PyPluginUnavailableBackend
from .utils import get_c_dtype_long

__all__ = [
    'PyPluginCuda',
    'CudaCompileFlags', 'CudaOptions'
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
    from numpy.typing import NDArray

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

            self.planes_functions += self.planes_functions[:-1] * (3 - len(self.planes_functions))

        if TYPE_CHECKING:
            def __call__(
                self, src: NDArray[Any], dst: NDArray[Any], *args: Any,
                kernel_size: tuple[int, int] = ..., block_size: tuple[int, int] = ..., shared_mem: int = ...
            ) -> Any:
                ...
        else:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return self.function(*args, **kwargs)

        def __getitem__(self, plane: int | None) -> CudaKernelFunction:
            if plane is None:
                return self.function

            return self.planes_functions[plane]

    class CudaKernelFunctions:
        def __init__(self, **kwargs: Any) -> None:
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

        cuda_options: CudaOptions = CudaOptions()
        cuda_flags: CudaCompileFlags = CudaCompileFlags()

        kernel_kwargs: dict[str, Any]

        kernel: CudaKernelFunctions

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

            cuda_path, cuda_functions = self.cuda_kernel
            if isinstance(cuda_functions, str):
                cuda_functions = [cuda_functions]

            if not isinstance(cuda_path, Path):
                cuda_path = Path(cuda_path)

            cuda_path = cuda_path.absolute().resolve()

            if cuda_path.exists():
                cuda_kernel_code = cuda_path.read_text()
            elif isinstance(self.cuda_kernel[0], str):
                cuda_kernel_code = self.cuda_kernel[0]

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
                name: RawKernel(code=cuda_kernel_code, name=name, **raw_kernel_kwargs) for name in cuda_functions
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

            self.kernel = CudaKernelFunctions(**{
                name: _wrap_kernel_function(
                    def_kernel_size, (block_x, block_y), def_shared_mem, function
                ) for name, function in self.kernel_functions.items()
            })

    this_backend.set_available(True)
except BaseException as e:
    this_backend.set_available(False, e)

    class PyPluginCuda(PyPluginUnavailableBackend[FD_T]):  # type: ignore
        backend = this_backend
