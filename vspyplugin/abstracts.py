from __future__ import annotations

from enum import IntEnum
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Mapping, TypeAlias, TypeVar, Union, cast, overload

import vapoursynth as vs

from .types import SupportsIndexing, SupportsKeysAndGetItem

__all__ = [
    'ProcessMode',
    'PyPluginBackendBase',
    'FD_T', 'DT_T'
]

F = TypeVar('F')
FVD = TypeVar('FVD', bound=Callable[..., vs.VideoNode])
FD_T = TypeVar('FD_T', bound=Any | SupportsKeysAndGetItem[str, object] | None)
DT_T = TypeVar('DT_T')

if TYPE_CHECKING:
    from .base import PyPlugin as CLS_T
else:
    CLS_T = TypeVar('CLS_T', bound=Generic)

PassthroughC = Callable[[F], F]


class ProcessModeBase:
    Any_T = Callable[
        [Any, DT_T | SupportsIndexing[DT_T], DT_T | SupportsIndexing[DT_T], vs.VideoFrame, int | None, int], None
    ]
    SingleSrcIPP_T = Callable[[Any, DT_T, DT_T, vs.VideoFrame, int, int], None]
    MultiSrcIPP_T = Callable[[Any, SupportsIndexing[DT_T], DT_T, vs.VideoFrame, int, int], None]
    SingleSrcIPF_T = Callable[[Any, DT_T, SupportsIndexing[DT_T], vs.VideoFrame, int], None]
    MultiSrcIPF_T = Callable[[Any, SupportsIndexing[DT_T], SupportsIndexing[DT_T], vs.VideoFrame, int], None]


class ProcessMode(ProcessModeBase, IntEnum):
    Any = -1
    SingleSrcIPP = 0
    MultiSrcIPP = 1
    SingleSrcIPF = 2
    MultiSrcIPF = 3


ALL_PMODES = Union[
    ProcessMode.Any_T[DT_T],
    ProcessMode.SingleSrcIPP_T[DT_T], ProcessMode.MultiSrcIPP_T[DT_T],
    ProcessMode.SingleSrcIPF_T[DT_T], ProcessMode.MultiSrcIPF_T[DT_T]
]


class PyPluginBackendOverloadedDict(dict[str, Any]):
    def __setitem__(self, key: str, value: Any) -> None:
        overloaded = getattr(value, '__pybackend_overload__', False)

        if overloaded:
            mode: ProcessMode = value.__pybackend_mode__

            if mode is ProcessMode.Any:
                for pmode in ProcessMode:
                    if pmode is ProcessMode.Any:
                        continue

                    mode_key = self.get_key(pmode)

                    if self.get(mode_key, None) is None:
                        if mode_key.endswith('IPF'):
                            value = partial(value, plane=None)

                        super().__setitem__(mode_key, value)

                return

            key = self.get_key(mode)

        super().__setitem__(key, value)

    @staticmethod
    def get_key(mode: ProcessMode) -> str:
        return f'process_{mode.name}'


class PyPluginBackendMeta(type):
    @classmethod
    def __prepare__(metacls, name: str, bases: tuple[type, ...], /, **kwargs: Any) -> Mapping[str, object]:
        return PyPluginBackendOverloadedDict(
            process_SingleSrcIPP=None, process_MultiSrcIPP=None,
            process_SingleSrcIPF=None, process_MultiSrcIPF=None,
            **kwargs
        )


class PyPluginBackendBase(Generic[DT_T], metaclass=PyPluginBackendMeta):
    DT: TypeAlias = DT_T  # type: ignore
    DTL: TypeAlias = SupportsIndexing[DT_T]  # type: ignore
    DTA: TypeAlias = DT_T | SupportsIndexing[DT_T]  # type: ignore

    process_SingleSrcIPP: ProcessMode.SingleSrcIPP_T[DT_T] | None
    process_MultiSrcIPP: ProcessMode.MultiSrcIPP_T[DT_T] | None
    process_SingleSrcIPF: ProcessMode.SingleSrcIPF_T[DT_T] | None
    process_MultiSrcIPF: ProcessMode.MultiSrcIPF_T[DT_T] | None

    @overload
    @staticmethod
    def process(mode: Literal[ProcessMode.Any], /) -> PassthroughC[ProcessMode.Any_T[DT_T]]:
        ...

    @overload
    @staticmethod
    def process(mode: Literal[ProcessMode.SingleSrcIPP], /) -> PassthroughC[ProcessMode.SingleSrcIPP_T[DT_T]]:
        ...

    @overload
    @staticmethod
    def process(mode: Literal[ProcessMode.MultiSrcIPP], /) -> PassthroughC[ProcessMode.MultiSrcIPP_T[DT_T]]:
        ...

    @overload
    @staticmethod
    def process(mode: Literal[ProcessMode.SingleSrcIPF], /) -> PassthroughC[ProcessMode.SingleSrcIPF_T[DT_T]]:
        ...

    @overload
    @staticmethod
    def process(mode: Literal[ProcessMode.MultiSrcIPF], /) -> PassthroughC[ProcessMode.MultiSrcIPF_T[DT_T]]:
        ...

    @overload
    @staticmethod
    def process(func: ProcessMode.Any_T[DT_T], /) -> ProcessMode.Any_T[DT_T]:
        ...

    @overload
    @staticmethod
    def process(func: None, /) -> PassthroughC[PassthroughC[ProcessMode.Any_T[DT_T]]]:
        ...

    @staticmethod  # type: ignore
    def process(mode_or_func: ProcessMode | ALL_PMODES[DT_T] | None = None, /) -> (
        PassthroughC[ALL_PMODES[DT_T]] | ProcessMode.Any_T[DT_T]
    ):
        if mode_or_func is None:
            return PyPluginBackendBase.process  # type: ignore
        elif not isinstance(mode_or_func, ProcessMode):
            return PyPluginBackendBase.process(ProcessMode.Any)(mode_or_func)  # type: ignore

        def _wrapper(func: Callable[..., None]) -> Callable[..., None]:
            func.__dict__.update(
                __pybackend_overload__=True,
                __pybackend_mode__=mode_or_func
            )

            return func

        return _wrapper

    @staticmethod
    def ensure_output(func: FVD) -> FVD:
        @wraps(func)
        def _wrapper(self: CLS_T[FD_T], *args: Any, **kwargs: Any) -> Any:
            return self.options.ensure_output(self, func(self, *args, **kwargs))

        return cast(FVD, _wrapper)
