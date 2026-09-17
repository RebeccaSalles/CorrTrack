from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Cython is required to build extensions. Install with `pip install cython`.") from exc

import numpy as np

extensions = [
    Extension(
        "candidate_kernels",
        ["candidate_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp", "-O3", "-march=native", "-ffast-math"],
        # (2026-07-30) -march=native -ffast-math lets gcc auto-vectorize the
        # new distance_corr_sketch_proxy_cy's sin()/cos() calls into glibc's
        # SIMD vector-math variants (libmvec), which need an explicit -lm
        # link (glibc >= 2.22 ships libmvec inside libm) -- without it the
        # extension fails to import with "undefined symbol: _ZGVdN4v_sin".
        extra_link_args=["-fopenmp", "-lm"],
    ),
    Extension(
        "sketch_kernels",
        ["sketch_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
    Extension(
        "partition_kernels",
        ["partition_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "competitor_kernels",
        ["competitor_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "monitor_kernels",
        ["monitor_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="corrtrack_cython_kernels",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
