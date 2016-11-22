import os
import sys
from setuptools import setup, find_packages

if sys.platform == "darwin":
    os.environ["CC"] = "gcc-4.8"
    os.environ["CXX"] = "g++-4.8"

packages = find_packages(exclude=['tests', 'tests._openmp', 'benchmark'])

setup(
    name='loky',
    version='0.1.dev0',
    packages=packages,
    zip_safe=False,
    license='BSD',
)
