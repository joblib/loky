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

MKL_MODULE_NAME = 'mkl_rt'
OMP_MODULE_NAME_GNU = 'gomp'
OMP_MODULE_NAME_INTEL = 'iomp'
OPENBLAS_MODULE_NAME = 'openblas'

_MODULE_NAME = None


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
                stderr=None)
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
# if a match is found, it set the global variable _MODULE_NAME to the full path
# of this library, to be loaded with ctypes.CDLL.
def check_module(info, size, data):
    global _MODULE_NAME
    # recast the name of the module as a string
    desired_module = ctypes.cast(data, ctypes.c_char_p).value.decode('utf-8')

    # Get the name of the current library
    info = ctypes.cast(info, ctypes.POINTER(dl_phdr_info))
    module_name = info.contents.dlpi_name

    # If the current library is the one we are looking for, set the global
    # variable with the desired path and return 1 to stop `dl_iterate_phdr`.
    if module_name:
        module_name = module_name.decode("utf-8")
        if module_name.find(desired_module) >= 0:
            _MODULE_NAME = module_name
            return 1
    return 0


class _CLibsWrapper:
    # Wrapper around classic C-library for scientific computations to set and
    # get the maximal number they can

    def __init__(self):
        self._load_openblas()
        self._load_omp()
        self._load_mkl()

    def limit_threads_clibs(self, max_threads_per_process):

        msg = ("max_threads_per_process should be an interger in "
               "limit_threads_clib")
        assert isinstance(max_threads_per_process, int), msg
        if os.name == "posix":
            self.openblas_set_num_threads(max_threads_per_process)
            self.mkl_set_num_threads(max_threads_per_process)
            self.omp_set_num_threads(max_threads_per_process)

    def get_thread_limits(self):
        return dict(
            OpenBLAS=self.openblas_get_max_threads(),
            OpenMP=self.omp_get_max_threads(),
            MKL=self.mkl_get_max_threads(),
        )

    def _load_openblas(self):
        self.lib_openblas = self._load_lib(OPENBLAS_MODULE_NAME,
                                           "openblas_info")

    def openblas_set_num_threads(self, num_threads):
        if self.lib_openblas is not None:
            try:
                self.lib_openblas.openblas_set_num_threads(num_threads)
            except OSError as e:
                return

    def openblas_get_max_threads(self):
        if self.lib_openblas is not None:
            try:
                return self.lib_openblas.openblas_get_num_threads()
            except OSError as e:
                pass
        return

    def _load_omp(self):
        self.lib_omp = self._load_lib(OMP_MODULE_NAME_GNU, "")
        if self.lib_omp is None:
            self.lib_omp = self._load_lib(OMP_MODULE_NAME_INTEL, "")

    def omp_set_num_threads(self, num_threads):

        if self.lib_omp is not None:
            try:
                self.lib_omp.omp_set_num_threads(num_threads)
            except OSError as e:
                return

    def omp_get_max_threads(self):

        if self.lib_omp is not None:
            try:
                return self.lib_omp.omp_get_max_threads()
            except OSError as e:
                pass
        return

    def _load_mkl(self):
        self.lib_mkl = self._load_lib(MKL_MODULE_NAME, "blas_mkl_info")

    def mkl_set_num_threads(self, num_threads):
        if self.lib_mkl is not None:
            try:
                self.lib_mkl.MKL_Set_Num_Threads(num_threads)
            except OSError as e:
                return

    def mkl_get_max_threads(self):
        if self.lib_mkl is not None:
            try:
                return self.lib_mkl.MKL_Get_Max_Threads()
            except OSError as e:
                pass
        return

    def _load_lib(self, module_name, lib_info):
        lib_name = find_library(module_name)
        if lib_name is not None:
            return ctypes.CDLL(lib_name, use_errno=True)
        return self._get_lib_from_numpy(module_name, lib_info)

    def _get_lib_from_numpy(self, module_name, lib_info):
        from glob import glob
        try:
            from numpy import __config__
            lib_info = getattr(__config__, lib_info, {})

            if lib_info:
                LIBRARY_GLOB = "{{}}/*{}*".format(module_name)
                for folder in lib_info['library_dirs']:
                    lib_names = glob(LIBRARY_GLOB.format(folder))
                    if len(lib_names) > 0:
                        return ctypes.CDLL(lib_names[0], use_errno=True)
        except ImportError:
            pass

        return self._get_lib_from_dynamic_libs(module_name)

    def _get_lib_from_dynamic_libs(self, module_name):

        global _MODULE_NAME
        _MODULE_NAME = None

        libc = ctypes.CDLL(find_library("c"))

        signature_cfunc = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
        c_check_module = signature_cfunc(check_module)

        data = ctypes.c_char_p(module_name.encode('utf-8'))
        res = libc.dl_iterate_phdr(c_check_module, data)
        if res == 1:
            return ctypes.CDLL(_MODULE_NAME)


_clibs_wrapper = None


def limit_threads_clib(max_threads_per_process):
    """Limit the number of threads available for OpenBLAS, MKL and OpenMP

    Set the maximal number of thread that can be used for these three libraries
    to `max_threads_per_process`. This function works on POSIX plateforms and
    can be used to change this limit dynamically.
    """
    global _clibs_wrapper
    if _clibs_wrapper is None:
        _clibs_wrapper = _CLibsWrapper()
    _clibs_wrapper.limit_threads_clibs(max_threads_per_process)


def get_thread_limits():
    """Return thread limit set for OpenBLAS, MKL and OpenMP

    Return a dictionary containing the maximal number of threads that can be
    used for these three library or None if this library is not available. The
    key of the dictionary are {'MKL', 'OpenBLAS', 'OpenMP'}
    """
    global _clibs_wrapper
    if _clibs_wrapper is None:
        _clibs_wrapper = _CLibsWrapper()
    return _clibs_wrapper.get_thread_limits()
