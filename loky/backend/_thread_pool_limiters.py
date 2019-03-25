
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

from .utils import _format_docstring


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


# List of the supported implementations. The items hold the prefix loaded
# shared object, the name of the internal_api to call, matching the
# MAP_API_TO_FUNC keys and the name of the user_api, in {"blas", "openmp"}.
SUPPORTED_IMPLEMENTATIONS = [
    {
        "filename_prefixes": ("libiomp", "libgomp", "libomp", "vcomp",),
        "internal_api": "openmp",
        "user_api": "openmp"
    },
    {
        "filename_prefixes": ("libopenblas",),
        "internal_api": "openblas",
        "user_api": "blas"
    },
    {
        "filename_prefixes": ("libmkl_rt", "mkl_rt",),
        "internal_api": "mkl",
        "user_api": "blas"
    },
]
ALL_USER_APIS = set(impl['user_api'] for impl in SUPPORTED_IMPLEMENTATIONS)
ALL_PREFIXES = [prefix for impl in SUPPORTED_IMPLEMENTATIONS
                for prefix in impl['filename_prefixes']]


# map a internal_api (openmp, openblas, mkl) to set and get functions
MAP_API_TO_FUNC = {
    "openmp": {
        "set": "omp_set_num_threads",
        "get": "omp_get_max_threads"},
    "openblas": {
        "set": "openblas_set_num_threads",
        "get": "openblas_get_num_threads"},
    "mkl": {
        "set": "MKL_Set_Num_Threads",
        "get": "MKL_Get_Max_Threads"},
}


class _ThreadPoolLibrariesWrapper:
    # Wrapper around classic C-libraries for scientific computations which use
    # thread-pools. Give access to functions to set and get the maximum number
    # of threads they are allowed to used for inner parallelism.

    cls_thread_locals = threading.local()

    def __init__(self):
        self._load()

    def _load(self, prefixes=None, user_api=None):
        if prefixes is None:
            prefixes = []
        if user_api is None:
            user_api = []
        if sys.platform == "darwin":
            return self._find_modules_with_clibs_dyld(
                prefixes=prefixes, user_api=user_api)
        elif sys.platform == "win32":
            return self._find_modules_with_enum_process_module_ex(
                prefixes=prefixes, user_api=user_api)
        else:
            return self._find_modules_with_clibs_dl_iterate_phdr(
                prefixes=prefixes, user_api=user_api)

    def starts_with_any(self, library_basename, filename_prefixes):
        for prefix in filename_prefixes:
            if library_basename.startswith(prefix):
                return prefix
        return None

    def _get_module_info_from_path(self, module_path):
        module_name = os.path.basename(module_path).lower()
        for info in SUPPORTED_IMPLEMENTATIONS:
            prefix = self.starts_with_any(module_name,
                                          info['filename_prefixes'])
            if prefix is not None:
                info = info.copy()
                info['prefix'] = prefix
                return info
        return None

    def _make_module(self, module_path, module_info):
        module_path = os.path.normpath(module_path)
        clib = ctypes.CDLL(module_path)
        internal_api = module_info['internal_api']
        set_func = getattr(clib, MAP_API_TO_FUNC[internal_api]['set'],
                           lambda n_thread: None)
        get_func = getattr(clib, MAP_API_TO_FUNC[internal_api]['get'],
                           lambda: None)
        module_info = module_info.copy()
        module_info.update(clib=clib, module_path=module_path,
                           set=set_func, get=get_func,
                           version=self.get_version(clib, internal_api))
        return module_info

    def get_limit(self, prefix, user_api, limits):
        if prefix in limits:
            return limits[prefix]
        if user_api in limits:
            return limits[user_api]
        return None

    def set_thread_limits(self, limits=None, user_api=None):
        """Limit maximal number of threads used by supported C-libraries.

        Return a list with the modules where the maximal number of threads
        available has been modified.
        """
        if isinstance(limits, int) or limits is None:
            if user_api in ('all', None):
                user_api = ALL_USER_APIS
            elif user_api in ALL_USER_APIS:
                user_api = (user_api,)
            else:
                raise ValueError("user_api must be either 'all', 'blas' or "
                                 "'openmp'. Got {} instead.".format(user_api))
            limits = {api: limits for api in user_api}
            prefixes = []
        else:
            if isinstance(limits, list):
                # This should be a list of module, for compatibility with
                # the result from get_thread_limits.
                limits = {module['prefix']: module['n_thread']
                          for module in limits}

            if not isinstance(limits, dict):
                raise TypeError("limits must either be an int, a dict or None."
                                " Got {} instead".format(type(limits)))

            # With a dictionary, can set both specific limit for given modules
            # and global limit for user_api. Fetch each separately.
            prefixes = [module for module in limits if module in ALL_PREFIXES]
            user_api = [module for module in limits if module in ALL_USER_APIS]

        report_threadpool_size = []
        modules = self._load(prefixes=prefixes, user_api=user_api)
        for module in modules:
            n_thread = self.get_limit(module['prefix'], module['user_api'],
                                      limits)
            if n_thread is not None:
                set_func = module['set']
                set_func(n_thread)

            # Store the module and remove un-necessary info
            module['n_thread'] = module['get']()
            del module['set'], module['get'], module['clib']
            report_threadpool_size.append(module)

        return report_threadpool_size

    def get_thread_limits(self):
        """Return maximal number of threads available for supported C-libraries
        """
        report_threadpool_size = []
        modules = self._load(user_api=ALL_USER_APIS)
        for module in modules:
            module['n_thread'] = module['get']()
            # Remove the wrapper for the module and its function
            del module['set'], module['get'], module['clib']
            report_threadpool_size.append(module)

        return report_threadpool_size

    def get_version(self, clib, internal_api):
        if internal_api == "mkl":
            return self.get_mkl_version(clib)
        elif internal_api == "openmp":
            # There is no way to get the version number programmatically in
            # OpenMP.
            return None
        elif internal_api == "openblas":
            return self.get_openblas_version(clib)
        else:
            raise NotImplementedError("Unsupported API {}"
                                      .format(internal_api))

    def get_openblas_version(self, openblas_clib):
        get_config = getattr(openblas_clib, "openblas_get_config")
        get_config.restype = ctypes.c_char_p
        config = get_config().split()
        if config[0] == b"OpenBLAS":
            return config[1].decode('utf-8')
        return None

    def get_mkl_version(self, mkl_clib):
        res = ctypes.create_string_buffer(200)
        mkl_clib.mkl_get_version_string(res, 200)

        version = res.value.decode('utf-8')
        return version.strip()

    def _include_modules(self, module_info, prefixes, user_api):
        return module_info is not None and (
            module_info['prefix'] in prefixes or
            module_info['user_api'] in user_api)

    def _find_modules_with_clibs_dl_iterate_phdr(self, prefixes, user_api):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developper @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """

        libc = self._get_libc()
        if not hasattr(libc, "dl_iterate_phdr"):  # pragma: no cover
            return []

        self.cls_thread_locals._modules = []

        # Callback function for `dl_iterate_phdr` which is called for every
        # module loaded in the current process until it returns 1.
        def match_module_callback(info, size, data):

            # Get the path of the current module
            module_path = info.contents.dlpi_name

            # If the current module is a supported module, store it in the
            # _modules, with a wrapper to its set and get functions and
            # extra information on the type of module it is.
            if module_path:
                module_path = module_path.decode("utf-8")
                module_info = self._get_module_info_from_path(module_path)
                if self._include_modules(module_info, prefixes, user_api):
                    self.cls_thread_locals._modules.append(
                        self._make_module(module_path, module_info))
            return 0

        c_func_signature = ctypes.CFUNCTYPE(
            ctypes.c_int,  # Return type
            ctypes.POINTER(dl_phdr_info), ctypes.c_size_t, ctypes.c_char_p)
        c_match_module_callback = c_func_signature(match_module_callback)

        data = ctypes.c_char_p(''.encode('utf-8'))
        libc.dl_iterate_phdr(c_match_module_callback, data)

        return self.cls_thread_locals._modules

    def _find_modules_with_clibs_dyld(self, prefixes, user_api):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on OSX system only
        """
        libc = self._get_libc()
        if not hasattr(libc, "_dyld_image_count"):  # pragma: no cover
            return []

        self.cls_thread_locals._modules = []

        n_dyld = libc._dyld_image_count()
        libc._dyld_get_image_name.restype = ctypes.c_char_p

        for i in range(n_dyld):
            module_path = ctypes.string_at(libc._dyld_get_image_name(i))
            module_path = module_path.decode("utf-8")

            # If the current module is a supported module, store it in the
            # _modules, with a wrapper to its set and get functions and
            # extra information on the type of module it is.
            module_info = self._get_module_info_from_path(module_path)
            if self._include_modules(module_info, prefixes, user_api):
                self.cls_thread_locals._modules.append(
                    self._make_module(module_path, module_info))

        return self.cls_thread_locals._modules

    def _find_modules_with_enum_process_module_ex(self, prefixes, user_api):
        """Loop through loaded libraries and return binders on supported ones

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
        if not h_process:  # pragma: no cover
            raise OSError('Could not open PID %s' % os.getpid())

        self.cls_thread_locals._modules = []
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

            # Loop through all the module headers and get the module path
            buf = ctypes.create_unicode_buffer(MAX_PATH)
            n_size = DWORD()
            for h_module in h_modules:
                if not ps_api.GetModuleFileNameExW(
                        h_process, h_module, ctypes.byref(buf),
                        ctypes.byref(n_size)):
                    raise OSError('GetModuleFileNameEx failed')
                module_path = buf.value

                # If the current module is a supported module, store it in the
                # _modules, with a wrapper to its set and get functions and
                # extra information on the type of module it is.
                module_info = self._get_module_info_from_path(module_path)
                if self._include_modules(module_info, prefixes, user_api):
                    self.cls_thread_locals._modules.append(
                        self._make_module(module_path, module_info))
        finally:
            kernel_32.CloseHandle(h_process)

        return self.cls_thread_locals._modules

    def _get_libc(self):
        if not hasattr(self, "libc"):
            libc_name = find_library("c")
            if libc_name is None:  # pragma: no cover
                self.libc = None
            self.libc = ctypes.CDLL(libc_name)

        return self.libc

    def _get_windll(self, dll_name):
        if not hasattr(self, dll_name):
            setattr(self, dll_name, ctypes.WinDLL("{}.dll".format(dll_name)))

        return getattr(self, dll_name)


_thread_pool_libraries_wrapper = None


def _get_wrapper(reload_modules=False):
    """Helper function to only create one wrapper per thread."""
    global _thread_pool_libraries_wrapper
    if _thread_pool_libraries_wrapper is None:
        _thread_pool_libraries_wrapper = _ThreadPoolLibrariesWrapper()
    if reload_modules:
        _thread_pool_libraries_wrapper._load()

    return _thread_pool_libraries_wrapper


@_format_docstring(ALL_PREFIXES=ALL_PREFIXES,
                   LIBRARIES=list(MAP_API_TO_FUNC.keys()))
def _set_thread_limits(limits=None, user_api=None, reload_modules=False):
    """Limit the maximal number of threads available for supported C-libraries.

    Set the maximal number of threads that can be used in thread pools used in
    the supported C-libraries to `limit`. This function works for libraries
    that are already loaded in the interpreter and can be changed dynamically.

    The `limits` parameter can be either an integer or a dict to specify the
    maximal number of thread that can be used in thread pools. If it is an
    integer, sets the maximum number of thread to `limits` for each C-lib
    selected by `user_api`. If it is a dictionary
    `{{supported_libraries: max_threads}}`, this function sets a custom maximum
    number of thread for each C-lib. If None, does not do anything.

    The `user_api` parameter selects particular APIs of C-libs to limit. Used
    only if `limits` is an int. If it is "all" or None, this function will
    apply to all supported C-libs. If it is "blas", it will limit only BLAS
    supported C-libs and if it is "openmp", only OpenMP supported C-libs will
    be limited. Note that the latter can affect the number of threads used by
    the BLAS C-libs if they rely on OpenMP.

    If `reload_modules` is `True`, first loop through the loaded libraries to
    ensure that this function is called on all the libraries that are used in
    this interpreter.

    Return a list with all the supported modules that have been found. Each
    module is represented by a dict with the following information:
      - 'filename-prefixes' : prefix of the specific implementation of this
            module. Possible values are {ALL_PREFIXES}.
      - 'internal_api': API for this module. Possible values are {LIBRARIES}.
      - 'module_path': path to the loaded module.
      - 'version': version of the library implemented (if available).
      - 'n_thread': current thread limit.
    """
    wrapper = _get_wrapper(reload_modules)
    return wrapper.set_thread_limits(limits, user_api)


@_format_docstring(ALL_PREFIXES=ALL_PREFIXES)
def get_thread_limits(reload_modules=False):
    """Return maximal thread number for threadpools in supported C-lib

    Return a dictionary containing the maximal number of threads that can be
    used in supported libraries or None when the library is not available. The
    key of the dictionary are {ALL_PREFIXES}.

    If `reload_modules` is `True`, first loop through the loaded libraries to
    ensure that this function is called on all available libraries.
    """
    wrapper = _get_wrapper(reload_modules)
    return wrapper.get_thread_limits()


class thread_pool_limits:
    """Change the maximal number of threads that can be used in thread pools.

    This class can be used either as a function (the construction of this
    object limits the number of threads) or as a context manager, in a `with`
    block.

    Parameters
    ----------
    limits : int, dict or None (default=None)
        Maximum number of thread that can be used in thread pools

        If int, sets the maximum number of thread to `limits` for each C-lib
        selected by `user_api`.

        If dict(supported_libraries: max_threads), sets a custom maximum number
        of thread for each C-lib.

        If None, does not do anything.

    user_api : string or None, optional (default="all")
        Subset of C-libs to limit. Used only if `limits` is an int

        "all" : limit all supported C-libs.

        "blas" : limit only BLAS supported C-libs.

        "openmp" : limit only OpenMP supported C-libs. It can affect the number
                   of threads used by the BLAS C-libs if they rely on OpenMP.

    .. versionadded:: 0.21

    """
    def __init__(self, limits=None, user_api=None):
        self._enabled = limits is not None
        if self._enabled:
            self.old_limits = get_thread_limits()
            _set_thread_limits(limits=limits, user_api=user_api)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.unregister()

    def unregister(self):
        if self._enabled:
            _set_thread_limits(limits=self.old_limits)
