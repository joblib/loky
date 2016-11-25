from setuptools import setup, find_packages

packages = find_packages(exclude=['tests', 'tests._openmp', 'benchmark'])

setup(
    name='loky',
    version='0.1.dev0',
    packages=packages,
    zip_safe=False,
    license='BSD',
)
