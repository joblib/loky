from setuptools import setup, find_packages

packages = find_packages(exclude=['tests', 'tests._openmp', 'benchmark'])

setup(
    name='loky',
    version='0.1.0',
    description=("A robust implementation of "
                 "concurrent.futures.ProcessPoolExecutor"),
    url='https://github.com/tommoral/loky/',
    author='Thomas Moreau',
    author_email='thomas.moreau.2010@gmail.com',
    packages=packages,
    zip_safe=False,
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
        'Topic :: Software Development :: Libraries',
    ],
    platforms='any'




)
