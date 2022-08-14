from __future__ import annotations

import vapoursynth as vs


def get_resolutions(clip: vs.VideoNode | vs.VideoFrame, strip: bool = False) -> tuple[tuple[int, int, int], ...]:
    assert clip.format

    chroma_res = (
        clip.width >> clip.format.subsampling_w, clip.height >> clip.format.subsampling_h
    )

    resolutions = (
        (0, clip.width, clip.height), (1, *chroma_res), (2, *chroma_res)
    )

    if clip.format.num_planes == 1:
        return resolutions[:1]

    return resolutions


def get_c_dtype_short(clip: vs.VideoNode) -> str:
    assert clip.format

    if clip.format.sample_type is vs.FLOAT:
        return get_c_dtype_long(clip)

    bps = clip.format.bytes_per_sample

    if bps == 1:
        return 'uchar'
    elif bps == 2:
        return 'ushort'
    elif bps == 4:
        return 'uint'

    raise RuntimeError


def get_c_dtype_long(clip: vs.VideoNode) -> str:
    assert clip.format

    bps = clip.format.bytes_per_sample

    if clip.format.sample_type is vs.FLOAT:
        if bps == 2:
            return 'half'
        return 'float'

    if bps == 1:
        return 'unsigned char'
    elif bps == 2:
        return 'unsigned short'
    elif bps == 4:
        return 'unsigned int'

    raise RuntimeError
