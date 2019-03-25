import ctypes
import pytest


from loky import cpu_count
from loky.backend import thread_pool_limits
from loky.backend._thread_pool_limiters import ALL_PREFIXES
from loky.backend._thread_pool_limiters import get_thread_limits
from loky.backend._thread_pool_limiters import _set_thread_limits

from .utils import with_parallel_sum, libopenblas_paths


def should_skip_module(module):
    return module['internal_api'] == "openblas" and module['version'] is None


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_thread_pool_limits(openblas_present, mkl_present, prefix):
    old_limits = get_thread_limits()

    prefix_found = len([1 for module in old_limits
                        if prefix == module['prefix']])
    old_limits = {clib['prefix']: clib['n_thread'] for clib in old_limits}

    if not prefix_found:
        if prefix == "libopenblas" and openblas_present:
            raise RuntimeError("Could not load the OpenBLAS prefix")
        elif "mkl_rt" in prefix and mkl_present:
            import numpy as np
            np.dot(np.ones(1000), np.ones(1000))
            old_limits = get_thread_limits()
            if old_limits[prefix] is None:
                raise RuntimeError("Could not load the MKL prefix")
        else:
            pytest.skip("Need {} support".format(prefix))

    new_limits = _set_thread_limits(limits={prefix: 1})
    new_limits = {clib['prefix']: clib['n_thread'] for clib in new_limits}
    assert new_limits[prefix] == 1

    thread_pool_limits(limits={prefix: 3})
    new_limits = get_thread_limits()
    new_limits = {clib['prefix']: clib['n_thread'] for clib in new_limits}
    assert new_limits[prefix] in (3, cpu_count(), cpu_count() // 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    new_limits = {clib['prefix']: clib['n_thread'] for clib in new_limits}
    assert new_limits[prefix] == old_limits[prefix]


@pytest.mark.parametrize("user_api", ("all", None, "blas", "openmp"))
def test_set_thread_limits_apis(user_api):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if user_api in ("all", None):
        api_modules = ('blas', 'openmp')
    else:
        api_modules = (user_api,)

    old_limits = get_thread_limits()
    old_limits = {clib['prefix']: clib['n_thread'] for clib in old_limits}

    new_limits = _set_thread_limits(limits=1, user_api=user_api)
    for module in new_limits:
        if should_skip_module(module):
            continue
        if module['user_api'] in api_modules:
            assert module['n_thread'] == 1

    thread_pool_limits(limits=3, user_api=user_api)
    new_limits = get_thread_limits()
    for module in new_limits:
        if should_skip_module(module):
            continue
        if module['user_api'] in api_modules:
            assert module['n_thread'] in (3, cpu_count(), cpu_count() // 2)

    thread_pool_limits(limits=old_limits)
    new_limits = get_thread_limits()
    for module in new_limits:
        assert module['n_thread'] == old_limits[module['prefix']]


def test_set_thread_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments

    with pytest.raises(ValueError,
                       match="user_api must be either 'all', 'blas' "
                             "or 'openmp'"):
        thread_pool_limits(limits=1, user_api="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int, a dict or None"):
        thread_pool_limits(limits=(1, 2, 3))


@pytest.mark.parametrize("user_api", (None, "all", "blas", "openmp"))
def test_thread_limit_context(user_api):
    # Tests the thread limits context manager

    if user_api in [None, "all"]:
        apis = ('blas', 'openmp')
    else:
        apis = (user_api,)

    old_limits = get_thread_limits()
    old_limits = {clib['prefix']: clib['n_thread'] for clib in old_limits}

    with thread_pool_limits(limits=None, user_api=user_api):
        limits = get_thread_limits()
        limits = {clib['prefix']: clib['n_thread'] for clib in limits}
        assert limits == old_limits

    with thread_pool_limits(limits=1, user_api=user_api):
        limits = get_thread_limits()

        for module in limits:
            if should_skip_module(module):
                continue
            elif module['user_api'] in apis:
                assert module['n_thread'] == 1
            else:
                assert module['n_thread'] == old_limits[module['prefix']]

    limits = get_thread_limits()
    limits = {clib['prefix']: clib['n_thread'] for clib in limits}
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
