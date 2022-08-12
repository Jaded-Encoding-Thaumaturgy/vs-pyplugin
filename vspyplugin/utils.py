import vapoursynth as vs


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
