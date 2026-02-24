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
    ),
    Extension(
        "sketch_kernels",
        ["sketch_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "partition_kernels",
        ["partition_kernels.pyx"],
        include_dirs=[np.get_include()],
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
