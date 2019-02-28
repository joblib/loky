import cython
cimport openmp
from cython.parallel import prange
from libc.stdlib cimport malloc, free


def parallel_sum(int n):
    cdef long n_sum = 0
    cdef int i, num_threads

    for i in prange(n, nogil=True):
        num_threads = openmp.omp_get_num_threads()
        n_sum += i

    assert n_sum == (n - 1) * n / 2

    return num_threads
