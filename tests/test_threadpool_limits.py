import os
import pytest


from loky import cpu_count
from loky.backend import thread_pool_limits
from loky.backend._thread_pool_limiters import _CLibsWrapper
from loky.backend._thread_pool_limiters import NOT_ACCESSIBLE
from loky.backend._thread_pool_limiters import get_thread_limits
from loky.backend._thread_pool_limiters import _set_thread_limits

from .utils import with_parallel_sum, with_numpy


def should_skip_module(module):
    return module['name'] == "openblas" and module['version'] == NOT_ACCESSIBLE


@pytest.mark.parametrize("library", _CLibsWrapper.SUPPORTED_CLIBS)
def test_thread_pool_limits(openblas_test_noskip, mkl_win32_test_noskip,
                            library):
    old_limits = get_thread_limits()
    old_limits = {clib['name']: clib['n_thread'] for clib in old_limits}

    if library not in old_limits:
        if library == "openblas" and openblas_test_noskip:
            raise RuntimeError("Could not load the OpenBLAS library")
        elif library == "mkl_win32" and mkl_win32_test_noskip:
            import numpy as np
            np.dot(np.ones(1000), np.ones(1000))
            old_limits = get_thread_limits()
            if old_limits[library] is None:
                raise RuntimeError("Could not load the MKL library")
        else:
            pytest.skip("Need {} support".format(library))

    new_limits = _set_thread_limits(limits={library: 1})
    new_limits = {clib['name']: clib['n_thread'] for clib in new_limits}
    assert new_limits[library] == 1

    thread_pool_limits(limits={library: 3})
    new_limits = get_thread_limits()
    new_limits = {clib['name']: clib['n_thread'] for clib in new_limits}
    assert new_limits[library] in (3, cpu_count(), cpu_count() / 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    new_limits = {clib['name']: clib['n_thread'] for clib in new_limits}
    assert new_limits[library] == old_limits[library]


@pytest.mark.parametrize("subset", ("all", "blas", "openmp"))
def test_set_thread_limits_subset(subset):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if subset == "all":
        apis = list(_CLibsWrapper.SUPPORTED_CLIBS.keys())
    elif subset == "blas":
        apis = ["openblas", "mkl", "mkl_win32"]
    elif subset == "openmp":
        apis = list(c for c in _CLibsWrapper.SUPPORTED_CLIBS if "openmp" in c)

    old_limits = get_thread_limits()
    old_limits = {clib['name']: clib['n_thread'] for clib in old_limits}

    new_limits = _set_thread_limits(limits=1, subset=subset)
    for module in new_limits:
        if module['api'] in apis and not should_skip_module(module):
            assert module['n_thread'] == 1

    thread_pool_limits(limits=3, subset=subset)
    new_limits = get_thread_limits()
    for module in new_limits:
        if module['api'] in apis and not should_skip_module(module):
            assert module['n_thread'] in (3, cpu_count(), cpu_count() / 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    for module in new_limits:
        assert module['n_thread'] == old_limits[module['name']]


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
    old_limits = {clib['name']: clib['n_thread'] for clib in old_limits}

    with thread_pool_limits(limits=None, subset=subset):
        limits = get_thread_limits()
        limits = {clib['name']: clib['n_thread'] for clib in limits}
        assert limits == old_limits

    with thread_pool_limits(limits=1, subset=subset):
        limits = get_thread_limits()
        limits = {clib['name']: clib['n_thread'] for clib in limits
                  if not should_skip_module(clib)}

        for clib in limits:
            if old_limits[clib] is None:
                assert limits[clib] is None
            elif clib in subset_clibs:
                assert limits[clib] == 1
            else:
                assert limits[clib] == old_limits[clib]

    limits = get_thread_limits()
    limits = {clib['name']: clib['n_thread'] for clib in limits}
    assert limits == old_limits


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


@with_numpy
def test_shipped_openblas():
    import ctypes
    import numpy as np
    from glob import glob
    libopenblas_patterns = [os.path.join(np.__path__[0], ".libs",
                                         "libopenblas*")]
    try:
        import scipy

        libopenblas_patterns += [os.path.join(scipy.__path__[0], ".libs",
                                              "libopenblas*")]
    except ImportError:
        pass
    libopenblas = [ctypes.CDLL(path) for pattern in libopenblas_patterns
                   for path in glob(pattern)]

    old_limits = [blas.openblas_get_num_threads() for blas in libopenblas]

    with thread_pool_limits(1):
        assert np.all([blas.openblas_get_num_threads() == 1
                       for blas in libopenblas])

    assert np.all([blas.openblas_get_num_threads() == l
                   for blas, l in zip(libopenblas, old_limits)])
