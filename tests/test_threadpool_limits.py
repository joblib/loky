import ctypes
import pytest


from loky import cpu_count
from loky.backend import thread_pool_limits
from loky.backend._thread_pool_limiters import SUPPORTED_IMPLEMENTATION
from loky.backend._thread_pool_limiters import get_thread_limits
from loky.backend._thread_pool_limiters import _set_thread_limits

from .utils import with_parallel_sum, libopenblas_paths


def should_skip_module(module):
    return module['name'] == "openblas" and module['version'] is None


@pytest.mark.parametrize("library", SUPPORTED_IMPLEMENTATION)
def test_thread_pool_limits(openblas_present, mkl_present, library):
    old_limits = get_thread_limits()
    old_limits = {clib['name']: clib['n_thread'] for clib in old_limits}

    if library not in old_limits:
        if library == "openblas" and openblas_present:
            raise RuntimeError("Could not load the OpenBLAS library")
        elif library == "mkl" and mkl_present:
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
    assert new_limits[library] in (3, cpu_count(), cpu_count() // 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    new_limits = {clib['name']: clib['n_thread'] for clib in new_limits}
    assert new_limits[library] == old_limits[library]


@pytest.mark.parametrize("apis", ("all", "blas", "openmp"))
def test_set_thread_limits_apis(apis):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if apis == "all":
        api_modules = list(SUPPORTED_IMPLEMENTATION.keys())
    elif apis == "blas":
        api_modules = ["openblas", "mkl"]
    elif apis == "openmp":
        api_modules = list(c for c in SUPPORTED_IMPLEMENTATION
                           if "openmp" in c)

    old_limits = get_thread_limits()
    old_limits = {clib['name']: clib['n_thread'] for clib in old_limits}

    new_limits = _set_thread_limits(limits=1, apis=apis)
    for module in new_limits:
        if module['library'] in api_modules and not should_skip_module(module):
            assert module['n_thread'] == 1

    thread_pool_limits(limits=3, apis=apis)
    new_limits = get_thread_limits()
    for module in new_limits:
        if module['library'] in api_modules and not should_skip_module(module):
            assert module['n_thread'] in (3, cpu_count(), cpu_count() // 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    for module in new_limits:
        assert module['n_thread'] == old_limits[module['name']]


def test_set_thread_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments

    with pytest.raises(ValueError,
                       match="apis must be either 'all', 'blas' "
                             "or 'openmp'"):
        thread_pool_limits(limits=1, apis="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int, a dict or None"):
        thread_pool_limits(limits=(1, 2, 3))


@pytest.mark.parametrize("apis", (None, "all", "blas", "openmp"))
def test_thread_limit_context(apis):
    # Tests the thread limits context manager

    if apis in [None, "all"]:
        apis_clibs = list(SUPPORTED_IMPLEMENTATION.keys())
    elif apis == "blas":
        apis_clibs = ["openblas", "mkl"]
    elif apis == "openmp":
        apis_clibs = list(c for c in SUPPORTED_IMPLEMENTATION
                          if "openmp" in c)

    old_limits = get_thread_limits()
    old_limits = {clib['name']: clib['n_thread'] for clib in old_limits}

    with thread_pool_limits(limits=None, apis=apis):
        limits = get_thread_limits()
        limits = {clib['name']: clib['n_thread'] for clib in limits}
        assert limits == old_limits

    with thread_pool_limits(limits=1, apis=apis):
        limits = get_thread_limits()
        limits = {clib['name']: clib['n_thread'] for clib in limits
                  if not should_skip_module(clib)}

        for clib in limits:
            if old_limits[clib] is None:
                assert limits[clib] is None
            elif clib in apis_clibs:
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
        assert parallel_sum(100) in (n_threads, cpu_count(), cpu_count() // 2)
    assert parallel_sum(100) == old_num_threads


def test_shipped_openblas():

    libopenblas = [ctypes.CDLL(path) for path in libopenblas_paths]

    old_limits = [blas.openblas_get_num_threads() for blas in libopenblas]

    with thread_pool_limits(1):
        assert all([blas.openblas_get_num_threads() == 1
                    for blas in libopenblas])

    assert all([blas.openblas_get_num_threads() == l
                for blas, l in zip(libopenblas, old_limits)])


@pytest.mark.skipif(len(libopenblas_paths) < 2,
                    reason="need at least 2 shipped openblas library")
def test_multiple_shipped_openblas():

    libopenblas = [ctypes.CDLL(path) for path in libopenblas_paths]

    old_limits = [blas.openblas_get_num_threads() for blas in libopenblas]

    with thread_pool_limits(1):
        assert all([blas.openblas_get_num_threads() == 1
                    for blas in libopenblas])

    assert all([blas.openblas_get_num_threads() == l
                for blas, l in zip(libopenblas, old_limits)])
