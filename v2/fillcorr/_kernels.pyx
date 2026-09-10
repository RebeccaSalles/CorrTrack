# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernel for FilCorr's Parseval correlation.

Compiled on the first import through pyximport (see
`backends._try_load_cython_kernel`). When Cython is not installed, the cython
backend falls back to the numpy version.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.complex128_t cplx_t


def parseval_corr_batch(np.ndarray[cplx_t, ndim=2] WX,
                        np.ndarray[cplx_t, ndim=2] WY):
    """Batched Parseval correlation: returns an (N,) float64.

    `WX`, `WY`: contiguous (N, B) complex128. NaN when one of the filtered
    windows has zero variance.
    """
    cdef Py_ssize_t n = WX.shape[0]
    cdef Py_ssize_t b = WX.shape[1]
    cdef np.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i, k
    cdef double sxx, syy, sxy
    cdef double xr, xi, yr, yi
    for i in range(n):
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(b):
            xr = WX[i, k].real
            xi = WX[i, k].imag
            yr = WY[i, k].real
            yi = WY[i, k].imag
            sxx += xr * xr + xi * xi
            syy += yr * yr + yi * yi
            sxy += xr * yr + xi * yi          # Re(conj(Wx) · Wy)
        if sxx > 0.0 and syy > 0.0:
            out[i] = sxy / sqrt(sxx * syy)
        else:
            out[i] = float("nan")
    return out
