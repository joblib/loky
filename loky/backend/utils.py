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

###########################################################################
# Code to load C-library that relies on thread pools and limit the maximal
# number of thread used.

# Structure to cast the the info on dynamically loaded library. See
# https://linux.die.net/man/3/dl_iterate_phdr for more details.
UINT_SYSTEM = ctypes.c_uint64 if sys.maxsize > 2**32 else ctypes.c_uint32
UINT_HALF_SYSTEM = ctypes.c_uint32 if sys.maxsize > 2**32 else ctypes.c_uint16


class dl_phdr_info(ctypes.Structure):
    _fields_ = [
        ("dlpi_addr",  UINT_SYSTEM),       # Base address of object
        ("dlpi_name",  ctypes.c_char_p),   # path to the library
        ("dlpi_phdr",  ctypes.c_void_p),   # pointer on dlpi_headers
        ("dlpi_phnum",  UINT_HALF_SYSTEM)  # number of element in dlpi_phdr
        ]


# This function is called on each dynamically loaded library attached to our
# process. It checks if any of this library matches the name given in data and
# if a match is found, it set the thread local variable _module_path to the
# full path of this library, to be loaded with ctypes.CDLL.
def match_module_callback(info, size, module_name):
    global _thread_locals

    # recast the name of the module as a string
    module_name = ctypes.string_at(module_name).decode('utf-8')

    # Get the name of the current library
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
    SUPPORTED_CLIB = {
        "openblas": (
            "libopenblas", "openblas_set_num_threads",
            "openblas_get_num_threads"),
        "openmp_intel": (
            "libiomp", "omp_set_num_threads", "omp_get_max_threads"),
        "openmp_gnu": (
            "libgomp", "omp_set_num_threads", "omp_get_max_threads"),
        "openmp_win32": (
            "vcomp", "omp_set_num_threads", "omp_get_max_threads"),
        "mkl": (
            "libmkl_rt", "MKL_Set_Num_Threads", "MKL_Get_Max_Threads")
    }

    def __init__(self):
        self._load()

    def _load(self):
        for clib, (module_name, _, _) in self.SUPPORTED_CLIB.items():
            if not hasattr(self, clib):
                setattr(self, clib, self._load_lib(module_name))

    def limit_threads_clibs(self, max_threads_per_process):
        """Limit the maximal number of threads used by supported C-library"""
        msg = ("max_threads_per_process should be an interger. Got {}"
               .format(max_threads_per_process))
        assert isinstance(max_threads_per_process, int), msg

        dynamic_threadpool_size = {}
        for clib, (_, _set, _) in self.SUPPORTED_CLIB.items():
            module = getattr(self, clib, None)
            if module is not None:
                _set = getattr(module, _set)
                _set(max_threads_per_process)
                dynamic_threadpool_size[clib] = True
            else:
                dynamic_threadpool_size[clib] = False
        return dynamic_threadpool_size

    def get_thread_limits(self):
        limits = {}
        for clib, (_, _, _get) in self.SUPPORTED_CLIB.items():
            module = getattr(self, clib, None)
            if module is not None:
                _get = getattr(module, _get)
                limits[clib] = _get()
            else:
                limits[clib] = None
        return limits

    def _load_lib(self, module_name):
        """Return a binder on module_name by looping through loaded libraries
        """
        if sys.platform == "darwin":
            return self._find_with_libc_dyld(module_name)
        elif sys.platform == "win32":
            return self._find_with_libc_enum_process_module_ex(module_name)
        return self._find_with_libc_dl_iterate_phdr(module_name)

    def _find_with_libc_dl_iterate_phdr(self, module_name):
        """Return a binder on module_name by looping through loaded libraries

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developper @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """
        global _thread_locals
        _thread_locals._module_path = None

        libc = self._get_libc()
        if not hasattr(libc, "dl_iterate_phdr"):
            return

        c_func_signature = ctypes.CFUNCTYPE(
            ctypes.c_int,  # Return type
            ctypes.POINTER(dl_phdr_info), ctypes.c_size_t, ctypes.c_char_p)
        c_match_module_callback = c_func_signature(match_module_callback)

        data = ctypes.c_char_p(module_name.encode('utf-8'))
        res = libc.dl_iterate_phdr(c_match_module_callback, data)
        if res == 1:
            return ctypes.CDLL(_thread_locals._module_path)

    def _find_with_libc_dyld(self, module_name):
        """Return a binder on module_name by looping through loaded libraries

        This function is expected to work on OSX system only
        """
        libc = self._get_libc()
        if not hasattr(libc, "_dyld_image_count"):
            return

        found_module_path = None

        n_dyld = libc._dyld_image_count()
        libc._dyld_get_image_name.restype = ctypes.c_char_p

        for i in range(n_dyld):
            module_path = ctypes.string_at(libc._dyld_get_image_name(i))
            module_path = module_path.decode("utf-8")
            if os.path.basename(module_path).startswith(module_name):
                found_module_path = module_path

        if found_module_path:
            return ctypes.CDLL(found_module_path)

    def _find_with_libc_enum_process_module_ex(self, module_name):
        """Return a binder on module_name by looping through loaded libraries

        This function is expected to work on windows system only.
        This code is adapted from code by Philipp Hagemeister @phihag available
        at https://stackoverflow.com/questions/17474574
        """
        from ctypes.wintypes import DWORD, HMODULE, MAX_PATH

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        LIST_MODULES_ALL = 0x03

        Psapi = self._get_windll('Psapi')
        Kernel32 = self._get_windll('kernel32')

        hProcess = Kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            False, os.getpid())
        if not hProcess:
            raise OSError('Could not open PID %s' % os.getpid())

        found_module_path = None
        try:
            buf_count = 256
            needed = DWORD()
            # Grow the buffer until it becomes large enough to hold all the
            # module headers
            while True:
                buf = (HMODULE * buf_count)()
                buf_size = ctypes.sizeof(buf)
                if not Psapi.EnumProcessModulesEx(
                        hProcess, ctypes.byref(buf), buf_size,
                        ctypes.byref(needed), LIST_MODULES_ALL):
                    raise OSError('EnumProcessModulesEx failed')
                if buf_size >= needed.value:
                    break
                buf_count = needed.value // (buf_size // buf_count)

            count = needed.value // (buf_size // buf_count)
            hModules = map(HMODULE, buf[:count])

            # Loop through all the module headers and get the module file name
            buf = ctypes.create_unicode_buffer(MAX_PATH)
            nSize = DWORD()
            for hModule in hModules:
                if not Psapi.GetModuleFileNameExW(
                        hProcess, hModule, ctypes.byref(buf),
                        ctypes.byref(nSize)):
                    raise OSError('GetModuleFileNameEx failed')
                module_path = buf.value
                if os.path.basename(module_path).startswith(module_name):
                    found_module_path = module_path
        finally:
            Kernel32.CloseHandle(hProcess)

        if found_module_path:
            return ctypes.CDLL(found_module_path)

    def _get_libc(self):
        if not hasattr(self, "libc"):
            libc_name = find_library("c")
            if libc_name is None:
                self.libc = None
            self.libc = ctypes.CDLL(libc_name)

        return self.libc

    def _get_windll(self, dll_name):
        if not hasattr(self, dll_name):
            setattr(self, dll_name, ctypes.WinDLL("{}.dll".format(dll_name)))

        return getattr(self, dll_name)


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
