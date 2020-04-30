import os
import re
import shutil
from setuptools import setup, find_packages
from distutils.command.clean import clean as Clean

packages = find_packages(exclude=['tests', 'tests._openmp', 'benchmark'])


# Function to parse __version__ in `loky`
def find_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'loky', '__init__.py'), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}

setup(
    name='loky',
    version=find_version(),
    description=("A robust implementation of "
                 "concurrent.futures.ProcessPoolExecutor"),
    long_description=open('README.md', 'rb').read().decode('utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/joblib/loky/',
    author='Thomas Moreau',
    author_email='thomas.moreau.2010@gmail.com',
    packages=packages,
    zip_safe=False,
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
    cmdclass=cmdclass,
    platforms='any',
    install_requires=['cloudpickle'],
    tests_require=['pytest', 'psutil'],
)
