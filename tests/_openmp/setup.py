import os
import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


if sys.platform == "darwin":
    os.environ["CC"] = "gcc-4.8"
    os.environ["CXX"] = "g++-4.8"


ext_modules = [
    Extension(
        "parallel_sum",
        ["parallel_sum.pyx"],
        extra_compile_args=["-ffast-math", "-fopenmp"],
        extra_link_args=['-fopenmp'],
        )
]

setup(
    name='_openmp',
    ext_modules=cythonize(ext_modules),
)
