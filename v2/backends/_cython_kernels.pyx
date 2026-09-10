# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython kernels for the hot primitives (Pearson, projection).

Compiled on demand (see backends/cython.py). Works on contiguous double
arrays; the caller guarantees contiguity.
"""

import numpy as np

from libc.math cimport NAN, sqrt


cdef double _pearson(double[:] x, double[:] y) noexcept nogil:
    cdef Py_ssize_t n = x.shape[0], i
    cdef double mx = 0.0, my = 0.0, cov = 0.0, sxx = 0.0, syy = 0.0, dx, dy, den
    if n == 0:
        return NAN
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        cov += dx * dy
        sxx += dx * dx
        syy += dy * dy
    den = sqrt(sxx * syy)
    if den == 0.0:
        return NAN
    return cov / den


def pearson(x, y):
    cdef double[:] xv = np.ascontiguousarray(x, dtype=np.double)
    cdef double[:] yv = np.ascontiguousarray(y, dtype=np.double)
    return _pearson(xv, yv)


def correlate(pairs):
    cdef list out = []
    for x, y in pairs:
        out.append(pearson(x, y))
    return out


def project(matrix, values):
    cdef double[:, :] R = np.ascontiguousarray(matrix, dtype=np.double)
    cdef double[:] v = np.ascontiguousarray(values, dtype=np.double)
    cdef Py_ssize_t nv = R.shape[0], w = R.shape[1], i, j
    out = np.empty(nv, dtype=np.double)
    cdef double[:] o = out
    cdef double acc
    for i in range(nv):
        acc = 0.0
        for j in range(w):
            acc += R[i, j] * v[j]
        o[i] = acc
    return out
