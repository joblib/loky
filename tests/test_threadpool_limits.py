import re
import ctypes
import pytest


from loky import cpu_count
from loky.backend import threadpool_limits
from loky.backend._thread_pool_limiters import get_threadpool_limits
from loky.backend._thread_pool_limiters import _set_threadpool_limits
from loky.backend._thread_pool_limiters import ALL_PREFIXES, ALL_USER_APIS

from .utils import with_check_openmp_n_threads, libopenblas_paths


def should_skip_module(module):
    # Possible bug in getting maximum number of threads with OpenBLAS < 0.2.16
    # and OpenBLAS does not expose its version before 0.3.4.
    return module['internal_api'] == "openblas" and module['version'] is None


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_threadpool_limits(openblas_present, mkl_present, prefix):
    old_limits = get_threadpool_limits()

    prefix_found = len([1 for module in old_limits
                        if prefix == module['prefix']])
    old_limits = {clib['prefix']: clib['n_thread'] for clib in old_limits}

    if not prefix_found:
        have_mkl = len({'mkl_rt', 'libmkl_rt'}.intersection(old_limits)) > 0
        if "mkl_rt" in prefix and mkl_present and not have_mkl:
            raise RuntimeError("Could not load the MKL prefix")
        elif prefix == "libopenblas" and openblas_present:
            raise RuntimeError("Could not load the OpenBLAS prefix")
        else:
            pytest.skip("Need {} support".format(prefix))

    new_limits = _set_threadpool_limits(limits={prefix: 1})
    new_limits = {clib['prefix']: clib['n_thread'] for clib in new_limits}
    assert new_limits[prefix] == 1

    threadpool_limits(limits={prefix: 3})
    new_limits = get_threadpool_limits()
    new_limits = {clib['prefix']: clib['n_thread'] for clib in new_limits}
    assert new_limits[prefix] in (3, cpu_count(), cpu_count() // 2)

    threadpool_limits(limits=old_limits)
    new_limits = get_threadpool_limits()
    new_limits = {clib['prefix']: clib['n_thread'] for clib in new_limits}
    assert new_limits[prefix] == old_limits[prefix]


@pytest.mark.parametrize("user_api", (None, "blas", "openmp"))
def test_set_threadpool_limits_apis(user_api):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if user_api is None:
        api_modules = ('blas', 'openmp')
    else:
        api_modules = (user_api,)

    old_limits = get_threadpool_limits()
    old_limits = {clib['prefix']: clib['n_thread'] for clib in old_limits}

    new_limits = _set_threadpool_limits(limits=1, user_api=user_api)
    for module in new_limits:
        if should_skip_module(module):
            continue
        if module['user_api'] in api_modules:
            assert module['n_thread'] == 1

    threadpool_limits(limits=3, user_api=user_api)
    new_limits = get_threadpool_limits()
    for module in new_limits:
        if should_skip_module(module):
            continue
        if module['user_api'] in api_modules:
            assert module['n_thread'] in (3, cpu_count(), cpu_count() // 2)

    threadpool_limits(limits=old_limits)
    new_limits = get_threadpool_limits()
    for module in new_limits:
        assert module['n_thread'] == old_limits[module['prefix']]


def test_set_threadpool_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments

    match = re.escape("user_api must be either in {} or None."
                      .format(ALL_USER_APIS))
    with pytest.raises(ValueError, match=match):
        threadpool_limits(limits=1, user_api="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int, a dict or None"):
        threadpool_limits(limits=(1, 2, 3))


@pytest.mark.parametrize("user_api", (None, "blas", "openmp"))
def test_thread_limit_context(user_api):
    # Tests the thread limits context manager

    if user_api is None:
        apis = ('blas', 'openmp')
    else:
        apis = (user_api,)

    old_limits = get_threadpool_limits()
    old_limits = {clib['prefix']: clib['n_thread'] for clib in old_limits}

    with threadpool_limits(limits=None, user_api=user_api):
        limits = get_threadpool_limits()
        limits = {clib['prefix']: clib['n_thread'] for clib in limits}
        assert limits == old_limits

    with threadpool_limits(limits=1, user_api=user_api):
        limits = get_threadpool_limits()

        for module in limits:
            if should_skip_module(module):
                continue
            elif module['user_api'] in apis:
                assert module['n_thread'] == 1
            else:
                assert module['n_thread'] == old_limits[module['prefix']]

    limits = get_threadpool_limits()
    limits = {clib['prefix']: clib['n_thread'] for clib in limits}
    assert limits == old_limits


@with_check_openmp_n_threads
@pytest.mark.parametrize('n_threads', [1, 2, 4])
def test_openmp_limit_num_threads(n_threads):
    # checks that OpenMP effectively uses the number of threads requested by
    # the context manager

    from ._openmp_test_helper import check_openmp_n_threads

    old_num_threads = check_openmp_n_threads(100)

    with threadpool_limits(limits=n_threads):
        assert check_openmp_n_threads(100) in (n_threads, cpu_count(),
                                               cpu_count() // 2)
    assert check_openmp_n_threads(100) == old_num_threads


def test_shipped_openblas():

    libopenblas = [ctypes.CDLL(path) for path in libopenblas_paths]

    old_limits = [blas.openblas_get_num_threads() for blas in libopenblas]

    with threadpool_limits(1):
        assert all([blas.openblas_get_num_threads() == 1
                    for blas in libopenblas])

    assert all([blas.openblas_get_num_threads() == l
                for blas, l in zip(libopenblas, old_limits)])


@pytest.mark.skipif(len(libopenblas_paths) < 2,
                    reason="need at least 2 shipped openblas library")
def test_multiple_shipped_openblas():

    libopenblas = [ctypes.CDLL(path) for path in libopenblas_paths]

    old_limits = [blas.openblas_get_num_threads() for blas in libopenblas]

    with threadpool_limits(1):
        assert all([blas.openblas_get_num_threads() == 1
                    for blas in libopenblas])

    assert all([blas.openblas_get_num_threads() == l
                for blas, l in zip(libopenblas, old_limits)])
