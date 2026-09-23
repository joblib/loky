"""Helper module to test OpenMP support on Continuous Integration"""

import os
import sys

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

if sys.platform == "darwin":
    extra_compile_args = ["-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
elif sys.platform == "win32":
    extra_compile_args = ["/openmp"]
    extra_link_args = None
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "parallel_sum",
        ["parallel_sum.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="_openmp_test_helper",
    ext_modules=cythonize(ext_modules),
)
