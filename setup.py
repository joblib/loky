import re
import sys
import glob
import setuptools
from distutils import sysconfig
from setuptools import setup, find_packages


if sys.platform == 'win32':  # Windows
    macros = dict()
    libraries = ['ws2_32']
elif sys.platform.startswith('darwin'):  # macOS
    macros = dict(
        HAVE_SEM_OPEN=1,
        HAVE_SEM_TIMEDWAIT=0,
        HAVE_FD_TRANSFER=1,
        HAVE_BROKEN_SEM_GETVALUE=1
    )
    libraries = []
elif sys.platform.startswith('cygwin'):  # Cygwin
    macros = dict(
        HAVE_SEM_OPEN=1,
        HAVE_SEM_TIMEDWAIT=1,
        HAVE_FD_TRANSFER=0,
        HAVE_BROKEN_SEM_UNLINK=1
    )
    libraries = []
elif sys.platform in ('freebsd4', 'freebsd5', 'freebsd6'):
    # FreeBSD's P1003.1b semaphore support is very experimental
    # and has many known problems. (as of June 2008)
    macros = dict(                  # FreeBSD 4-6
        HAVE_SEM_OPEN=0,
        HAVE_SEM_TIMEDWAIT=0,
        HAVE_FD_TRANSFER=1,
    )
    libraries = []
elif re.match('^(gnukfreebsd(8|9|10|11)|freebsd(7|8|9|0))', sys.platform):
    macros = dict(                  # FreeBSD 7+ and GNU/kFreeBSD 8+
        HAVE_SEM_OPEN=bool(
            sysconfig.get_config_var('HAVE_SEM_OPEN') and not
            bool(sysconfig.get_config_var('POSIX_SEMAPHORES_NOT_ENABLED'))
        ),
        HAVE_SEM_TIMEDWAIT=1,
        HAVE_FD_TRANSFER=1,
    )
    libraries = []
elif sys.platform.startswith('openbsd'):
    macros = dict(                  # OpenBSD
        HAVE_SEM_OPEN=0,            # Not implemented
        HAVE_SEM_TIMEDWAIT=0,
        HAVE_FD_TRANSFER=1,
    )
    libraries = []
else:                                   # Linux and other unices
    macros = dict(
        HAVE_SEM_OPEN=1,
        HAVE_SEM_TIMEDWAIT=1,
        HAVE_FD_TRANSFER=1,
    )
    libraries = ['rt']

if sys.platform == 'win32':
    multiprocessing_srcs = [
        'Modules/_semaphore/multiprocessing.c',
        'Modules/_semaphore/semaphore.c',
        'Modules/_semaphore/win32_functions.c',
    ]
else:
    multiprocessing_srcs = [
        'Modules/_semaphore/multiprocessing.c',
    ]

    if macros.get('HAVE_SEM_OPEN', False):
        multiprocessing_srcs.append('Modules/_semaphore/semaphore.c')

if sys.version_info[0] == 3:
    macros['PY3'] = 1


setup(name='loky',
      version='0.1.dev0',
      packages=find_packages(),
      )


def _is_build_command(argv=sys.argv, cmds=('install', 'build', 'bdist')):
    for arg in argv:
        if arg.startswith(cmds):
            return arg


def run_setup(with_extensions=True):
    extensions = []
    print(list(macros.items()))
    if with_extensions:
        extensions = [
            setuptools.Extension(
                '_semaphore',
                sources=multiprocessing_srcs,
                define_macros=list(macros.items()),
                libraries=libraries,
                include_dirs=['Modules/_semaphore'],
                depends=glob.glob('Modules/_semaphore/*.h') + ['setup.py'],
            ),
        ]
        if sys.platform == 'win32':
            extensions.append(
                setuptools.Extension(
                    '_winapi',
                    sources=multiprocessing_srcs,
                    define_macros=list(macros.items()),
                    libraries=libraries,
                    include_dirs=['Modules/_semaphore'],
                    depends=glob.glob('Modules/_semaphore/*.h') + ['setup.py'],
                ),
            )
    packages = setuptools.find_packages(exclude=['ez_setup', 't', 't.*'])
    setuptools.setup(
        name='Rpool',
        packages=packages,
        ext_modules=extensions,
        zip_safe=False,
        license='BSD'
    )

try:
    run_setup(False)
except BaseException:
    if _is_build_command(sys.argv):
        import traceback
        print(BUILD_WARNING % '\n'.join(traceback.format_stack()))
        run_setup(False)
    else:
        raise
