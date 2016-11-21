import cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free

def parallel_sum(int N):
    cdef double Ysum = 0
    cdef int i
    cdef double* X = <double *>malloc(N*cython.sizeof(double))

    for i in prange(N, nogil=True):
        Ysum += X[i]

    free(X)
