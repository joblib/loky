import os
import sys
import errno
import signal
import ctypes
import warnings
import threading
import subprocess
from ctypes.util import find_library

try:
    import psutil
except ImportError:
    psutil = None

_thread_locals = threading.local() 
_thread_locals._module_path = None


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def recursive_terminate(process, use_psutil=True):
    if use_psutil and psutil is not None:
        _recursive_terminate_with_psutil(process)
    else:
        _recursive_terminate_without_psutil(process)


def _recursive_terminate_with_psutil(process, retries=5):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return

    # Kill the children in reverse order to avoid killing the parents before
    # the children in cases where there are more processes nested.
    for child in children[::-1]:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    process.terminate()
    process.join()


def _recursive_terminate_without_psutil(process):
    """Terminate a process and its descendants.
    """
    try:
        _recursive_terminate(process.pid)
    except OSError as e:
        warnings.warn("Failed to kill subprocesses on this platform. Please"
                      "install psutil: https://github.com/giampaolo/psutil")
        # In case we cannot introspect the children, we fall back to the
        # classic Process.terminate.
        process.terminate()
    process.join()


def _recursive_terminate(pid):
    """Recursively kill the descendants of a process before killing it.
    """

    if sys.platform == "win32":
        # On windows, the taskkill function with option `/T` terminate a given
        # process pid and its children.
        try:
            subprocess.check_output(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # In windows, taskkill return 1 for permission denied and 128, 255
            # for no process found.
            if e.returncode not in [1, 128, 255]:
                raise
            elif e.returncode == 1:
                # Try to kill the process without its descendants if taskkill
                # was denied permission. If this fails too, with an error
                # different from process not found, let the top level function
                # raise a warning and retry to kill the process.
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as e:
                    if e.errno != errno.ESRCH:
                        raise

    else:
        try:
            children_pids = subprocess.check_output(
                ["pgrep", "-P", str(pid)],
                stderr=None
            )
        except subprocess.CalledProcessError as e:
            # `ps` returns 1 when no child process has been found
            if e.returncode == 1:
                children_pids = b''
            else:
                raise

        # Decode the result, split the cpid and remove the trailing line
        children_pids = children_pids.decode().split('\n')[:-1]
        for cpid in children_pids:
            cpid = int(cpid)
            _recursive_terminate(cpid)

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            # if OSError is raised with [Errno 3] no such process, the process
            # is already terminated, else, raise the error and let the top
            # level function raise a warning and retry to kill the process.
            if e.errno != errno.ESRCH:
                raise


##############################################################################
# The following code is derived from code by Intel developper @anton-malakhov
# available at https://github.com/IntelPython/smp
#
# Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
# license
#


# Structure to cast the the info on dynamically loaded library.
if sys.maxsize > 2**32:
    class dl_phdr_info(ctypes.Structure):
        _fields_ = [("dlpi_addr",  ctypes.c_uint64),
                    ("dlpi_name",  ctypes.c_char_p),
                    ("dlpi_phdr",  ctypes.c_void_p),
                    ("dlpi_phnum", ctypes.c_uint16)]
else:
    class dl_phdr_info(ctypes.Structure):
        _fields_ = [("dlpi_addr",  ctypes.c_uint32),
                    ("dlpi_name",  ctypes.c_char_p),
                    ("dlpi_phdr",  ctypes.c_void_p),
                    ("dlpi_phnum", ctypes.c_uint16)]


# This function is called on each dynamically loaded library attached to our
# process. It checks if any of this library matches the name given in data and
# if a match is found, it set the thread local variable _module_path to the
# full path of this library, to be loaded with ctypes.CDLL.
def match_module_callback(info, size, module_name):
    global _thread_locals
    # recast the name of the module as a string
    module_name = ctypes.cast(module_name, ctypes.c_char_p).value
    module_name = module_name.decode('utf-8')

    # Get the name of the current library
    info = ctypes.cast(info, ctypes.POINTER(dl_phdr_info))
    module_path = info.contents.dlpi_name

    # If the current library is the one we are looking for, set the global
    # variable with the desired path and return 1 to stop `dl_iterate_phdr`.
    if module_path:
        module_path = module_path.decode("utf-8")
        if os.path.basename(module_path).startswith(module_name):
            _thread_locals._module_path = module_path
            return 1
    return 0


class _CLibsWrapper:
    # Wrapper around classic C-library for scientific computations to set and
    # get the maximum number of threads they are allowed to used for inner
    # parallelism.

    # Here are the C-library supported on linux platforms, with their name and
    # the name of the library file which is looked-up.
    SUPPORTED_CLIB = [
        ("openblas", "openblas"),
        ("openmp_intel", "iomp"),
        ("openmp_gnu", "gomp"),
        ("mkl", "mkl_rt")
    ]

    def __init__(self):
        self._load()

    def _load(self):
        for clib, module_name in self.SUPPORTED_CLIB:
            if not hasattr(self, clib):
                setattr(self, clib, self._load_lib(module_name))

    def limit_threads_clibs(self, max_threads_per_process):

        msg = ("max_threads_per_process should be an interger. Got {}"
               .format(max_threads_per_process))
        assert isinstance(max_threads_per_process, int), msg

        dynamic_threadpool_size = {}
        for clib, _ in self.SUPPORTED_CLIB:
            try:
                _set = getattr(self, "{}_set_num_threads".format(clib))
                _set(max_threads_per_process)
                dynamic_threadpool_size[clib] = True
            except NotImplementedError:
                dynamic_threadpool_size[clib] = False
        return dynamic_threadpool_size

    def get_thread_limits(self):
        limits = {}
        for clib, _ in self.SUPPORTED_CLIB:
            try:
                _get = getattr(self, "{}_get_max_threads".format(clib))
                limits[clib] = _get()
            except NotImplementedError:
                limits[clib] = None
        return limits

    def openblas_set_num_threads(self, num_threads):
        if self.openblas is None:
            raise NotImplementedError("Could not find OpenBLAS library.")
        self.openblas.openblas_set_num_threads(num_threads)

    def openblas_get_max_threads(self):
        if self.openblas is None:
            raise NotImplementedError("Could not find OpenBLAS library.")
        return self.openblas.openblas_get_num_threads()

    def openmp_gnu_set_num_threads(self, num_threads):

        if self.openmp_gnu is None:
            raise NotImplementedError("Could not find OpenMP library")
        self.openmp_gnu.omp_set_num_threads(num_threads)

    def openmp_gnu_get_max_threads(self):
        if self.openmp_gnu is None:
            raise NotImplementedError("Could not find OpenMP library")
        return self.openmp_gnu.omp_get_max_threads()

    def openmp_intel_set_num_threads(self, num_threads):
        if self.openmp_intel is None:
            raise NotImplementedError("Could not find OpenMP library")
        self.openmp_intel.omp_set_num_threads(num_threads)

    def openmp_intel_get_max_threads(self):
        if self.openmp_intel is None:
            raise NotImplementedError("Could not find OpenMP library")
        return self.openmp_intel.omp_get_max_threads()

    def mkl_set_num_threads(self, num_threads):
        if self.mkl is None:
            raise NotImplementedError("Could not find MKL libray")
        self.mkl.MKL_Set_Num_Threads(num_threads)

    def mkl_get_max_threads(self):
        if self.mkl is None:
            raise NotImplementedError("Could not find MKL libray")
        return self.mkl.MKL_Get_Max_Threads()

    def _load_lib(self, module_name):
        lib_name = find_library(module_name)
        if lib_name is not None:
            return ctypes.CDLL(lib_name, use_errno=True)
        return self._find_with_libc_dl_iterate_phdr(module_name)

    def _find_with_libc_dl_iterate_phdr(self, module_name):

        global _thread_locals
        _thread_locals._module_path = None

        libc_name = find_library("c")
        if libc_name is None:
            return
        libc = ctypes.CDLL(libc_name)
        if not hasattr(libc, "dl_iterate_phdr"):
            return

        c_func_signature = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
        c_match_module_callback = c_func_signature(match_module_callback)

        module_name = "lib{}".format(module_name)
        data = ctypes.c_char_p(module_name.encode('utf-8'))
        res = libc.dl_iterate_phdr(c_match_module_callback, data)
        if res == 1:
            return ctypes.CDLL(_thread_locals._module_path)


_clibs_wrapper = None


def _get_wrapper():
    global _clibs_wrapper
    if _clibs_wrapper is None:
        _clibs_wrapper = _CLibsWrapper()
    return _clibs_wrapper


def limit_threads_clib(max_threads_per_process):
    """Limit the number of threads available for openblas, mkl and openmp

    Set the maximal number of thread that can be used for these three libraries
    to `max_threads_per_process`. This function works on POSIX plateforms and
    can be used to change this limit dynamically.

    Return a dict dynamic_threadpool_size containing pairs `('clib': boolean)`
    which are True if `clib` have been found and can be used to scale the
    maximal number of hreads dynamically.
    """
    return _get_wrapper().limit_threads_clibs(max_threads_per_process)


def get_thread_limits():
    """Return thread limit set for openblas, mkl and openmp

    Return a dictionary containing the maximal number of threads that can be
    used for these three library or None if this library is not available. The
    key of the dictionary are {'mkl', 'openblas', 'openmp_gnu', 'openmp_intel'}
    """
    return _get_wrapper().get_thread_limits()
