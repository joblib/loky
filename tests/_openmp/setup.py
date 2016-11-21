from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension(
        "parallel_sum",
        ["parallel_sum.pyx"],
        libraries=["m"],
        extra_compile_args=["-ffast-math", "-fopenmp"],
        extra_link_args=['-fopenmp'],
        )
]

setup(
    name='_openmp',
    ext_modules=cythonize(ext_modules),
)
