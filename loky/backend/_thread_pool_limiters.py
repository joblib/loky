
#############################################################################
# The following provides utilities to load C-libraries that relies on thread
# pools and limit the maximal number of thread that can be used.
#
#
import os
import sys
import ctypes
import threading
from ctypes.util import find_library

from ..backend.context import cpu_count


if sys.platform == "darwin":

    # On OSX, we can get a runtime error due to multiple OpenMP libraries
    # loaded simultaneously. This can happen for instance when calling BLAS
    # inside a prange. Setting the following environment variable allows
    # multiple OpenMP libraries to be loaded. It should not degrade
    # performances since we manually take care of potential over-subscription
    # performance issues, in sections of the code where nested OpenMP loops can
    # happen, by dynamically reconfiguring the inner OpenMP runtime to
    # temporarily disable it while under the scope of the outer OpenMP parallel
    # section.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


def openmp_num_threads(n_threads=None):
    """Determine the effective number of threads used for parallel OpenMP calls

    For n_threads=None, returns openmp.omp_get_max_threads().
    For n_threads > 0, use this as the maximal number of threads for parallel
    OpenMP calls and for n_threads < 0, use the maximal number of threads minus
    - n_threads + 1.
    Raise a ValueError for n_threads=0.
    """
    if n_threads == 0:
        raise ValueError("n_threads=0 is invalid")
    elif n_threads < 0:
        return max(1, cpu_count() + n_threads + 1)
    else:
        return n_threads


# Structure to cast the info on dynamically loaded library. See
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


class _CLibsWrapper:
    # Wrapper around classic C-libraries for scientific computations to set and
    # get the maximum number of threads they are allowed to used for inner
    # parallelism.

    # Supported C-libraries for this wrapper, index with their name. The items
    # hold the name of the library file and the functions to call.
    SUPPORTED_CLIBS = {
        "openmp_intel": (
            "libiomp", "omp_set_num_threads", "omp_get_max_threads"),
        "openmp_gnu": (
            "libgomp", "omp_set_num_threads", "omp_get_max_threads"),
        "openmp_llvm": (
            "libomp", "omp_set_num_threads", "omp_get_max_threads"),
        "openmp_win32": (
            "vcomp", "omp_set_num_threads", "omp_get_max_threads"),
        "openblas": (
            "libopenblas", "openblas_set_num_threads",
            "openblas_get_num_threads"),
        "mkl": (
            "libmkl_rt", "MKL_Set_Num_Threads", "MKL_Get_Max_Threads"),
        "mkl_win32": (
            "mkl_rt", "MKL_Set_Num_Threads", "MKL_Get_Max_Threads")}

    cls_thread_locals = threading.local()

    def __init__(self):
        self._load()

    def _load(self):
        for clib, (module_name, _, _) in self.SUPPORTED_CLIBS.items():
            setattr(self, clib, self._load_lib(module_name))

    def _unload(self):
        for clib, (module_name, _, _) in self.SUPPORTED_CLIBS.items():
            delattr(self, clib)

    def set_thread_limits(self, limits=None, subset=None):
        """Limit maximal number of threads used by supported C-libraries.

        Return a dict of pairs {clib: bool} where bool is True when clib can be
        dynamically scaled with this function.
        """
        if isinstance(limits, int) or limits is None:
            if subset in ("all", None):
                clibs = self.SUPPORTED_CLIBS.keys()
            elif subset == "blas":
                clibs = ("openblas", "mkl", "mkl_win32")
            elif subset == "openmp":
                clibs = (c for c in self.SUPPORTED_CLIBS if "openmp" in c)
            else:
                raise ValueError("subset must be either 'all', 'blas' or "
                                 "'openmp'. Got {} instead.".format(subset))
            limits = {clib: limits for clib in clibs}

        if not isinstance(limits, dict):
            raise TypeError("limits must either be an int, a dict or None. "
                            "Got {} instead".format(type(limits)))

        dynamic_threadpool_size = {}
        self._load()
        for clib, (_, _set_name, _) in self.SUPPORTED_CLIBS.items():
            if clib in limits:
                modules = getattr(self, clib, [])
                dynamic_threadpool_size[clib] = False
                for module in modules:
                    _set = getattr(module, _set_name)
                    if limits[clib] is not None:
                        _set(openmp_num_threads(limits[clib]))
                    dynamic_threadpool_size[clib] = True
            else:
                dynamic_threadpool_size[clib] = False
        self._unload()
        return dynamic_threadpool_size

    def get_thread_limits(self):
        """Return maximal number of threads available for supported C-libraries
        """
        limits = {}
        self._load()
        for clib, (_, _, _get_name) in self.SUPPORTED_CLIBS.items():
            limits[clib] = None
            modules = getattr(self, clib, [])
            for module in modules:
                _get = getattr(module, _get_name)
                # If multiple version of a library are present, return the max
                limit = limits[clib]
                limits[clib] = max(_get(), limit) if limit else _get()
        self._unload()
        return limits

    def get_openblas_version(self):
        modules = getattr(self, "openblas", [])
        for module in modules:
            get_config = getattr(module, "openblas_get_config")
            get_config.restype = ctypes.c_char_p
            config = get_config().split()
            if config[0] == b"OpenBLAS":
                return config[1].decode('utf-8')

    def _load_lib(self, module_name):
        """Return a binder on module_name by looping through loaded libraries
        """
        if sys.platform == "darwin":
            return self._find_with_clibs_dyld(module_name)
        elif sys.platform == "win32":
            return self._find_with_clibs_enum_process_module_ex(module_name)
        return self._find_with_clibs_dl_iterate_phdr(module_name)

    def _find_with_clibs_dl_iterate_phdr(self, module_name):
        """Return a binder on module_name by looping through loaded libraries

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developper @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """

        libc = self._get_libc()
        if not hasattr(libc, "dl_iterate_phdr"):
            return

        self.cls_thread_locals._module_paths = []

        # Callback function for `dl_iterate_phdr` which is called for every
        # module loaded in the current process until it returns 1.
        def match_module_callback(info, size, module_name):

            # recast the name of the module as a string
            module_name = ctypes.string_at(module_name).decode('utf-8')

            # Get the name of the current library
            module_path = info.contents.dlpi_name

            # If the current library is the one we are looking for, store the
            # path and return 0 to continue the loop in `dl_iterate_phdr`.
            if module_path:
                module_path = module_path.decode("utf-8")
                if os.path.basename(module_path).startswith(module_name):
                    self.cls_thread_locals._module_paths.append(module_path)
            return 0

        c_func_signature = ctypes.CFUNCTYPE(
            ctypes.c_int,  # Return type
            ctypes.POINTER(dl_phdr_info), ctypes.c_size_t, ctypes.c_char_p)
        c_match_module_callback = c_func_signature(match_module_callback)

        data = ctypes.c_char_p(module_name.encode('utf-8'))
        libc.dl_iterate_phdr(c_match_module_callback, data)

        return [ctypes.CDLL(path)
                for path in self.cls_thread_locals._module_paths]

    def _find_with_clibs_dyld(self, module_name):
        """Return a binder on module_name by looping through loaded libraries

        This function is expected to work on OSX system only
        """
        libc = self._get_libc()
        if not hasattr(libc, "_dyld_image_count"):
            return

        self.cls_thread_locals._module_paths = []

        n_dyld = libc._dyld_image_count()
        libc._dyld_get_image_name.restype = ctypes.c_char_p

        for i in range(n_dyld):
            module_path = ctypes.string_at(libc._dyld_get_image_name(i))
            module_path = module_path.decode("utf-8")
            if os.path.basename(module_path).startswith(module_name):
                self.cls_thread_locals.found_module_paths.append(module_path)

        return [ctypes.CDLL(path)
                for path in self.cls_thread_locals.found_module_paths]

    def _find_with_clibs_enum_process_module_ex(self, module_name):
        """Return a binder on module_name by looping through loaded libraries

        This function is expected to work on windows system only.
        This code is adapted from code by Philipp Hagemeister @phihag available
        at https://stackoverflow.com/questions/17474574
        """
        from ctypes.wintypes import DWORD, HMODULE, MAX_PATH

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        LIST_MODULES_ALL = 0x03

        ps_api = self._get_windll('Psapi')
        kernel_32 = self._get_windll('kernel32')

        h_process = kernel_32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            False, os.getpid())
        if not h_process:
            raise OSError('Could not open PID %s' % os.getpid())

        self.cls_thread_locals._module_paths = []
        try:
            buf_count = 256
            needed = DWORD()
            # Grow the buffer until it becomes large enough to hold all the
            # module headers
            while True:
                buf = (HMODULE * buf_count)()
                buf_size = ctypes.sizeof(buf)
                if not ps_api.EnumProcessModulesEx(
                        h_process, ctypes.byref(buf), buf_size,
                        ctypes.byref(needed), LIST_MODULES_ALL):
                    raise OSError('EnumProcessModulesEx failed')
                if buf_size >= needed.value:
                    break
                buf_count = needed.value // (buf_size // buf_count)

            count = needed.value // (buf_size // buf_count)
            h_modules = map(HMODULE, buf[:count])

            # Loop through all the module headers and get the module file name
            buf = ctypes.create_unicode_buffer(MAX_PATH)
            n_size = DWORD()
            for h_module in h_modules:
                if not ps_api.GetModuleFileNameExW(
                        h_process, h_module, ctypes.byref(buf),
                        ctypes.byref(n_size)):
                    raise OSError('GetModuleFileNameEx failed')
                module_path = buf.value
                module_basename = os.path.basename(module_path).lower()
                if module_basename.startswith(module_name):
                    self.cls_thread_locals.found_module_paths.append(
                        module_path)
        finally:
            kernel_32.CloseHandle(h_process)

        return [ctypes.CDLL(path)
                for path in self.cls_thread_locals.found_module_paths]

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


def _get_wrapper(reload_clib=False):
    """Helper function to only create one wrapper per thread."""
    global _clibs_wrapper
    if _clibs_wrapper is None:
        _clibs_wrapper = _CLibsWrapper()
    if reload_clib:
        _clibs_wrapper._load()

    return _clibs_wrapper


def _set_thread_limits(limits=None, subset=None, reload_clib=False):
    """Limit the number of threads available for threadpools in supported C-lib

    Set the maximal number of thread that can be used in thread pools used in
    the supported C-libraries to `max_threads_per_process`. This function works
    for libraries that are already loaded in the interpreter and can be changed
    dynamically.

    The `limits` parameter can be either an interger or a dict to specify the
    maximal number of thread that can be used in thread pools. If it is an
    integer, sets the maximum number of thread to `limits` for each C-lib
    selected by `subset`. If it is a dictionary
    `{supported_libraries: max_threads}`, this function sets a custom maximum
    number of thread for each C-lib. If None, does not do anything.

    The `subset` parameter select a subset of C-libs to limit. Used only if
    `limits` is an int. If it is "all" or None, this function will limit all
    supported C-libs. If it is "blas", it will limit only BLAS supported
    C-libs and if it is "openmp", only only OpenMP supported C-libs will be
    limited. Note that the latter can affect the number of threads used by the
    BLAS C-libs if they rely on OpenMP.

    If `reload_clib` is `True`, first loop through the loaded libraries to
    ensure that this function is called on all available libraries.

    Return a dict dynamic_threadpool_size containing pairs `('clib': boolean)`
    which are True if `clib` have been found and can be used to scale the
    maximal number of threads dynamically.
    """
    wrapper = _get_wrapper(reload_clib)
    return wrapper.set_thread_limits(limits, subset)


def get_thread_limits(reload_clib=False):
    """Return maximal thread number for threadpools in supported C-lib

    Return a dictionary containing the maximal number of threads that can be
    used in supported libraries or None when the library is not available. The
    key of the dictionary are {}.

    If `reload_clib` is `True`, first loop through the loaded libraries to
    ensure that this function is called on all available libraries.
    """.format(list(_CLibsWrapper.SUPPORTED_CLIBS.keys()))
    wrapper = _get_wrapper(reload_clib)
    return wrapper.get_thread_limits()


class thread_pool_limits:
    """Change the default number of threads used in thread pools.

    This class can be used either as a function (the construction of this
    object limits the number of threads) or as a context manager, in a `with`
    block.

    Parameters
    ----------
    limits : int or dict, (default=None)
        Maximum number of thread that can be used in thread pools

        If int, sets the maximum number of thread to `limits` for each C-lib
        selected by `subset`.

        If dict(supported_libraries: max_threads), sets a custom maximum number
        of thread for each C-lib.

        If None, does not do anything.

    subset : string or None, optional (default="all")
        Subset of C-libs to limit. Used only if `limits` is an int

        "all" : limit all supported C-libs.

        "blas" : limit only BLAS supported C-libs.

        "openmp" : limit only OpenMP supported C-libs. It can affect the number
                   of threads used by the BLAS C-libs if they rely on OpenMP.

    .. versionadded:: 0.21

    """
    def __init__(self, limits=None, subset=None):

        self.old_limits = get_thread_limits()
        _set_thread_limits(limits=limits, subset=subset)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.unregister()

    def unregister(self):
        _set_thread_limits(limits=self.old_limits)


def get_openblas_version(reload_clib=True):
    """Return the OpenBLAS version

    Parameters
    ----------
    reload_clib : bool, (default=True)
        If `reload_clib` is `True`, first loop through the loaded libraries to
        ensure that this function is called on all available libraries.

    Returns
    -------
    version : string or None
        None means OpenBLAS is not loaded or version < 0.3.4, since OpenBLAS
        did not expose it's verion before that.
    """
    wrapper = _get_wrapper(reload_clib)
    return wrapper.get_openblas_version()
