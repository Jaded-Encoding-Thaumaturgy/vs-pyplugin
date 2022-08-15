# cython: boundscheck=False, initializedcheck=False, language_level=3, nonecheck=False, overflowcheck=False, wraparound=False

cimport cython
from cython cimport view
from libc.math cimport fabs
from cython.parallel import prange


cdef int clamp(const int val, const int low, const int high) nogil:
    return min(max(val, low), high)


cpdef void sigma_filter(
    const float [:, ::view.contiguous] src, float [:, ::view.contiguous] dst, 
    const int radius, const float threshold
) nogil:
    cdef int height = src.shape[0]
    cdef int width = src.shape[1]

    cdef float center, val, acc
    cdef int count, x, y, i, j

    with nogil:
        for y in prange(height):
            for x in prange(width):
                center = src[y, x]

                acc = 0.0
                count = 0

                for j in prange(clamp(y - radius, 0, height - 1), clamp(y + radius, 0, height - 1) + 1):
                    for i in prange(clamp(x - radius, 0, width - 1), clamp(x + radius, 0, width - 1) + 1):
                        val = src[j, i]

                        if fabs(center - val) < threshold:
                            acc += val
                            count += 1
                
                dst[y, x] = acc / count