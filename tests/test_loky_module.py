import os
import shutil
from subprocess import check_output

import pytest

import loky
from loky import cpu_count
from loky.backend.utils import _CLibsWrapper
from loky.backend.utils import get_thread_limits
from loky.backend.utils import _set_thread_limits
from loky.backend.utils import thread_pool_limits
from loky.backend.utils import get_openblas_version

from .utils import with_parallel_sum


SKIP_OPENBLAS = get_openblas_version() is None


def test_version():
    assert hasattr(loky, '__version__'), (
        "There are no __version__ argument on the loky module")


def test_cpu_count():
    cpus = cpu_count()
    assert type(cpus) is int
    assert cpus >= 1


cpu_count_cmd = ("from loky.backend.context import cpu_count;"
                 "print(cpu_count())")


def test_cpu_count_affinity():
    if not hasattr(os, 'sched_getaffinity') or not hasattr(shutil, 'which'):
        pytest.skip()

    taskset_bin = shutil.which('taskset')
    python_bin = shutil.which('python')

    if taskset_bin is None or python_bin is None:
        raise pytest.skip()

    try:
        os.sched_getaffinity(0)
    except NotImplementedError:
        pytest.skip()

    res = check_output([taskset_bin, '-c', '0',
                        python_bin, '-c', cpu_count_cmd])

    assert res.strip().decode('utf-8') == '1'


def test_cpu_count_cfs_limit():
    if not hasattr(shutil, 'which'):
        pytest.skip()

    docker_bin = shutil.which('docker')
    if docker_bin is None:
        raise pytest.skip()

    loky_path = os.path.abspath(os.path.dirname(loky.__file__))

    # The following will always run using the Python 3.6 docker image.
    # We mount the loky source as /loky inside the container,
    # so it can be imported when running commands under /
    res = check_output([docker_bin, 'run', '--rm', '--cpus', '0.5',
                        '-v', '%s:/loky' % loky_path,
                        'python:3.6',
                        'python', '-c', cpu_count_cmd])

    assert res.strip().decode('utf-8') == '1'


@pytest.mark.parametrize("clib", _CLibsWrapper.SUPPORTED_CLIBS)
def test_thread_pool_limits(openblas_test_noskip, mkl_win32_test_noskip, clib):
    old_limits = get_thread_limits()

    if old_limits[clib] is None:
        if clib == "openblas" and openblas_test_noskip:
            raise RuntimeError("Could not load the OpenBLAS library")
        elif clib == "mkl_win32" and mkl_win32_test_noskip:
            import numpy as np
            np.dot(np.ones(1000), np.ones(1000))
            old_limits = get_thread_limits()
            if old_limits[clib] is None:
                raise RuntimeError("Could not load the MKL library")
        else:
            pytest.skip("Need {} support".format(clib))

    dynamic_scaling = _set_thread_limits(limits={clib: 1})
    assert get_thread_limits()[clib] == 1
    assert dynamic_scaling[clib]

    thread_pool_limits(limits={clib: 3})
    new_limits = get_thread_limits()
    assert new_limits[clib] in (3, cpu_count(), cpu_count() / 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    assert new_limits[clib] == old_limits[clib]


@pytest.mark.parametrize("subset", ("all", "blas", "openmp"))
def test_set_thread_limits_subset(subset):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if subset == "all":
        clibs = list(_CLibsWrapper.SUPPORTED_CLIBS.keys())
    elif subset == "blas":
        clibs = ["openblas", "mkl", "mkl_win32"]
    elif subset == "openmp":
        clibs = list(c for c in _CLibsWrapper.SUPPORTED_CLIBS if "openmp" in c)

    if SKIP_OPENBLAS and "openblas" in clibs:
        clibs.remove("openblas")

    old_limits = get_thread_limits()

    dynamic_scaling = _set_thread_limits(limits=1, subset=subset)
    new_limits = get_thread_limits()
    for clib in clibs:
        if old_limits[clib] is not None:
            assert new_limits[clib] == 1
            assert dynamic_scaling[clib]

    thread_pool_limits(limits=3, subset=subset)
    new_limits = get_thread_limits()
    for clib in clibs:
        if old_limits[clib] is not None:
            assert new_limits[clib] in (3, cpu_count(), cpu_count() / 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    for clib in clibs:
        if old_limits[clib] is not None:
            assert new_limits[clib] == old_limits[clib]


def test_set_thread_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments

    with pytest.raises(ValueError,
                       match="subset must be either 'all', 'blas' "
                             "or 'openmp'"):
        thread_pool_limits(limits=1, subset="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int, a dict"):
        thread_pool_limits(limits=(1, 2, 3))


@pytest.mark.parametrize("subset", (None, "all", "blas", "openmp"))
def test_thread_limit_context(subset):
    # Tests the thread limits context manager

    if subset in [None, "all"]:
        subset_clibs = list(_CLibsWrapper.SUPPORTED_CLIBS.keys())
    elif subset == "blas":
        subset_clibs = ["openblas", "mkl", "mkl_win32"]
    elif subset == "openmp":
        subset_clibs = list(c for c in _CLibsWrapper.SUPPORTED_CLIBS
                            if "openmp" in c)

    old_limits = get_thread_limits()

    with thread_pool_limits(limits=None, subset=subset):
        assert get_thread_limits() == old_limits

    with thread_pool_limits(limits=1, subset=subset):
        limits = get_thread_limits()
        if SKIP_OPENBLAS:
            del limits["openblas"]

        for clib in limits:
            if old_limits[clib] is None:
                assert limits[clib] is None
            elif clib in subset_clibs:
                assert limits[clib] == 1
            else:
                assert limits[clib] == old_limits[clib]

    assert get_thread_limits() == old_limits


@with_parallel_sum
@pytest.mark.parametrize('n_threads', [1, 2, 4])
def test_openmp_limit_num_threads(n_threads):
    # checks that OpenMP effectively uses the number of threads requested by
    # the context manager

    from ._openmp_test_helper.parallel_sum import parallel_sum

    old_num_threads = parallel_sum(100)

    with thread_pool_limits(limits=n_threads):
        assert parallel_sum(100) in (n_threads, cpu_count(), cpu_count() / 2)
    assert parallel_sum(100) == old_num_threads
