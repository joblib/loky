
#############################################################################
# The following provides utilities to load C-libraries that relies on thread
# pools and limit the maximal number of thread that can be used.
#
#
import os
import re
import sys
import ctypes
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


# List of the supported implementations. The items hold the prefix of loaded
# shared objects, the name of the internal_api to call, matching the
# MAP_API_TO_FUNC keys and the name of the user_api, in {"blas", "openmp"}.
SUPPORTED_IMPLEMENTATIONS = [
    {
        "user_api": "openmp",
        "internal_api": "openmp",
        "filename_prefixes": ("libiomp", "libgomp", "libomp", "vcomp",),
    },
    {
        "user_api": "blas",
        "internal_api": "openblas",
        "filename_prefixes": ("libopenblas",),
    },
    {
        "user_api": "blas",
        "internal_api": "mkl",
        "filename_prefixes": ("libmkl_rt", "mkl_rt",),
    },
]

# map a internal_api (openmp, openblas, mkl) to set and get functions
MAP_API_TO_FUNC = {
    "openmp": {
        "set_num_threads": "omp_set_num_threads",
        "get_num_threads": "omp_get_max_threads"},
    "openblas": {
        "set_num_threads": "openblas_set_num_threads",
        "get_num_threads": "openblas_get_num_threads"},
    "mkl": {
        "set_num_threads": "MKL_Set_Num_Threads",
        "get_num_threads": "MKL_Get_Max_Threads"},
}

# Helpers for the doc and test names
ALL_USER_APIS = set(impl['user_api'] for impl in SUPPORTED_IMPLEMENTATIONS)
ALL_PREFIXES = [prefix for impl in SUPPORTED_IMPLEMENTATIONS
                for prefix in impl['filename_prefixes']]
ALL_INTERNAL_APIS = list(MAP_API_TO_FUNC.keys())


def _get_limit(prefix, user_api, limits):
    if prefix in limits:
        return limits[prefix]
    if user_api in limits:
        return limits[user_api]
    return None


@_format_docstring(ALL_PREFIXES=ALL_PREFIXES, INTERNAL_APIS=ALL_INTERNAL_APIS)
def _set_threadpool_limits(limits=None, user_api=None):
    """Limit the maximal number of threads for threadpools in supported C-lib

    Set the maximal number of threads that can be used in thread pools used in
    the supported C-libraries to `limit`. This function works for libraries
    that are already loaded in the interpreter and can be changed dynamically.

    The `limits` parameter can be either an integer or a dict to specify the
    maximal number of thread that can be used in thread pools. If it is an
    integer, sets the maximum number of thread to `limits` for each C-lib
    selected by `user_api`. If it is a dictionary `{{key: max_threads}}`,
    this function sets a custom maximum number of thread for each `key` which
    can be either a `user_api` or a `prefix` for a specific library.
    If None, this function does not do anything.

    The `user_api` parameter selects particular APIs of C-libs to limit. Used
    only if `limits` is an int. If it is None, this function will apply to all
    supported C-libs. If it is "blas", it will limit only BLAS supported C-libs
    and if it is "openmp", only OpenMP supported C-libs will be limited. Note
    that the latter can affect the number of threads used by the BLAS C-libs if
    they rely on OpenMP.

    Return a list with all the supported modules that have been found. Each
    module is represented by a dict with the following information:
      - 'filename-prefixes' : possible prefixes for the given internal_api.
            Possible values are {ALL_PREFIXES}.
      - 'prefix' : prefix of the specific implementation of this module.
      - 'internal_api': internal API.s Possible values are {INTERNAL_APIS}.
      - 'module_path': path to the loaded module.
      - 'version': version of the library implemented (if available).
      - 'n_thread': current thread limit.
    """
    if isinstance(limits, int) or limits is None:
        if user_api is None:
            user_api = ALL_USER_APIS
        elif user_api in ALL_USER_APIS:
            user_api = (user_api,)
        else:
            raise ValueError("user_api must be either in {} or None. Got {} "
                             "instead.".format(ALL_USER_APIS, user_api))
        limits = {api: limits for api in user_api}
        prefixes = []
    else:
        if isinstance(limits, list):
            # This should be a list of module, for compatibility with
            # the result from get_threadpool_limits.
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
    modules = _load_modules(prefixes=prefixes, user_api=user_api)
    for module in modules:
        n_thread = _get_limit(module['prefix'], module['user_api'], limits)
        if n_thread is not None:
            set_func = module['set_num_threads']
            set_func(n_thread)

        # Store the module and remove un-necessary info
        module['n_thread'] = module['get_num_threads']()
        del module['set_num_threads'], module['get_num_threads']
        del module['clib']
        report_threadpool_size.append(module)

    return report_threadpool_size


@_format_docstring(ALL_PREFIXES=ALL_PREFIXES, INTERNAL_APIS=ALL_INTERNAL_APIS)
def get_threadpool_limits():
    """Return the maximal number of threads for threadpools in supported C-lib.

    Return a list with all the supported modules that have been found. Each
    module is represented by a dict with the following information:
      - 'filename-prefixes' : possible prefixes for the given internal_api.
            Possible values are {ALL_PREFIXES}.
      - 'prefix' : prefix of the specific implementation of this module.
      - 'internal_api': internal API. Possible values are {INTERNAL_APIS}.
      - 'module_path': path to the loaded module.
      - 'version': version of the library implemented (if available).
      - 'n_thread': current thread limit.
    """
    report_threadpool_size = []
    modules = _load_modules(user_api=ALL_USER_APIS)
    for module in modules:
        module['n_thread'] = module['get_num_threads']()
        # Remove the wrapper for the module and its function
        del module['set_num_threads'], module['get_num_threads']
        del module['clib']
        report_threadpool_size.append(module)

    return report_threadpool_size


def get_version(clib, internal_api):
    if internal_api == "mkl":
        return _get_mkl_version(clib)
    elif internal_api == "openmp":
        # There is no way to get the version number programmatically in
        # OpenMP.
        return None
    elif internal_api == "openblas":
        return _get_openblas_version(clib)
    else:
        raise NotImplementedError("Unsupported API {}".format(internal_api))


def _get_mkl_version(mkl_clib):
    """Return the MKL version
    """
    res = ctypes.create_string_buffer(200)
    mkl_clib.mkl_get_version_string(res, 200)

    version = res.value.decode('utf-8')
    group = re.search(r"Version ([^ ]+) ", version)
    if group is not None:
        version = group.groups()[0]
    return version.strip()


def _get_openblas_version(openblas_clib):
    """Return the OpenBLAS version

    None means OpenBLAS is not loaded or version < 0.3.4, since OpenBLAS
    did not expose its version before that.
    """
    get_config = getattr(openblas_clib, "openblas_get_config")
    get_config.restype = ctypes.c_char_p
    config = get_config().split()
    if config[0] == b"OpenBLAS":
        return config[1].decode('utf-8')
    return None


#################################################################
# Loading utilities for dynamically linked shared objects

def _load_modules(prefixes=None, user_api=None):
    """Loop through loaded libraries and return supported ones."""
    if prefixes is None:
        prefixes = []
    if user_api is None:
        user_api = []
    if sys.platform == "darwin":
        return _find_modules_with_clibs_dyld(
            prefixes=prefixes, user_api=user_api)
    elif sys.platform == "win32":
        return _find_modules_with_enum_process_module_ex(
            prefixes=prefixes, user_api=user_api)
    else:
        return _find_modules_with_clibs_dl_iterate_phdr(
            prefixes=prefixes, user_api=user_api)


def _check_prefix(library_basename, filename_prefixes):
    """Return the prefix library_basename starts with or None if none matches
    """
    for prefix in filename_prefixes:
        if library_basename.startswith(prefix):
            return prefix
    return None


def _match_module(module_info, prefix, prefixes, user_api):
    """Return True if this module should be selected."""
    return prefix is not None and (prefix in prefixes or
                                   module_info['user_api'] in user_api)


def _make_module_info(module_path, module_info, prefix):
    """Make a dict with the information from the module."""
    module_path = os.path.normpath(module_path)
    clib = ctypes.CDLL(module_path)
    internal_api = module_info['internal_api']
    set_func = getattr(clib, MAP_API_TO_FUNC[internal_api]['set_num_threads'],
                       lambda n_thread: None)
    get_func = getattr(clib, MAP_API_TO_FUNC[internal_api]['get_num_threads'],
                       lambda: None)
    module_info = module_info.copy()
    module_info.update(clib=clib, module_path=module_path, prefix=prefix,
                       set_num_threads=set_func, get_num_threads=get_func,
                       version=get_version(clib, internal_api))
    return module_info


def _get_module_info_from_path(module_path, prefixes, user_api, modules):
    module_name = os.path.basename(module_path).lower()
    for info in SUPPORTED_IMPLEMENTATIONS:
        prefix = _check_prefix(module_name, info['filename_prefixes'])
        if _match_module(info, prefix, prefixes, user_api):
            modules.append(_make_module_info(module_path, info, prefix))


def _find_modules_with_clibs_dl_iterate_phdr(prefixes, user_api):
    """Loop through loaded libraries and return binders on supported ones

    This function is expected to work on POSIX system only.
    This code is adapted from code by Intel developper @anton-malakhov
    available at https://github.com/IntelPython/smp

    Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
    license
    """

    libc = _get_libc()
    if not hasattr(libc, "dl_iterate_phdr"):  # pragma: no cover
        return []

    _modules = []

    # Callback function for `dl_iterate_phdr` which is called for every
    # module loaded in the current process until it returns 1.
    def match_module_callback(info, size, data):

        # Get the path of the current module
        module_path = info.contents.dlpi_name
        if module_path:
            module_path = module_path.decode("utf-8")

            # Store the module in cls_thread_locals._module if it is
            # supported and selected
            _get_module_info_from_path(module_path, prefixes, user_api,
                                       _modules)
        return 0

    c_func_signature = ctypes.CFUNCTYPE(
        ctypes.c_int,  # Return type
        ctypes.POINTER(dl_phdr_info), ctypes.c_size_t, ctypes.c_char_p)
    c_match_module_callback = c_func_signature(match_module_callback)

    data = ctypes.c_char_p(''.encode('utf-8'))
    libc.dl_iterate_phdr(c_match_module_callback, data)

    return _modules


def _find_modules_with_clibs_dyld(prefixes, user_api):
    """Loop through loaded libraries and return binders on supported ones

    This function is expected to work on OSX system only
    """
    libc = _get_libc()
    if not hasattr(libc, "_dyld_image_count"):  # pragma: no cover
        return []

    _modules = []

    n_dyld = libc._dyld_image_count()
    libc._dyld_get_image_name.restype = ctypes.c_char_p

    for i in range(n_dyld):
        module_path = ctypes.string_at(libc._dyld_get_image_name(i))
        module_path = module_path.decode("utf-8")

        # Store the module in cls_thread_locals._module if it is
        # supported and selected
        _get_module_info_from_path(module_path, prefixes, user_api,
                                   _modules)

    return _modules


def _find_modules_with_enum_process_module_ex(prefixes, user_api):
    """Loop through loaded libraries and return binders on supported ones

    This function is expected to work on windows system only.
    This code is adapted from code by Philipp Hagemeister @phihag available
    at https://stackoverflow.com/questions/17474574
    """
    from ctypes.wintypes import DWORD, HMODULE, MAX_PATH

    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    LIST_MODULES_ALL = 0x03

    ps_api = _get_windll('Psapi')
    kernel_32 = _get_windll('kernel32')

    h_process = kernel_32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        False, os.getpid())
    if not h_process:  # pragma: no cover
        raise OSError('Could not open PID %s' % os.getpid())

    _modules = []
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

            # Get the path of the current module
            if not ps_api.GetModuleFileNameExW(
                    h_process, h_module, ctypes.byref(buf),
                    ctypes.byref(n_size)):
                raise OSError('GetModuleFileNameEx failed')
            module_path = buf.value

            # Store the module in cls_thread_locals._module if it is
            # supported and selected
            _get_module_info_from_path(module_path, prefixes, user_api,
                                       _modules)
    finally:
        kernel_32.CloseHandle(h_process)

    return _modules


def _get_libc():
    """Load the lib-C for unix systems."""
    libc_name = find_library("c")
    if libc_name is None:  # pragma: no cover
        return None
    return ctypes.CDLL(libc_name)


def _get_windll(dll_name):
    """Load a windows DLL"""
    return ctypes.WinDLL("{}.dll".format(dll_name))


class threadpool_limits:
    """Change the maximal number of threads that can be used in thread pools.

    This class can be used either as a function (the construction of this
    object limits the number of threads) or as a context manager, in a `with`
    block.

    Set the maximal number of threads that can be used in thread pools used in
    the supported C-libraries to `limit`. This function works for libraries
    that are already loaded in the interpreter and can be changed dynamically.

    The `limits` parameter can be either an integer or a dict to specify the
    maximal number of thread that can be used in thread pools. If it is an
    integer, sets the maximum number of thread to `limits` for each C-lib
    selected by `user_api`. If it is a dictionary `{{key: max_threads}}`,
    this function sets a custom maximum number of thread for each `key` which
    can be either a `user_api` or a `prefix` for a specific library.
    If None, this function does not do anything.

    The `user_api` parameter selects particular APIs of C-libs to limit. Used
    only if `limits` is an int. If it is None, this function will apply to all
    supported C-libs. If it is "blas", it will limit only BLAS supported C-libs
    and if it is "openmp", only OpenMP supported C-libs will be limited. Note
    that the latter can affect the number of threads used by the BLAS C-libs if
    they rely on OpenMP.
    """
    def __init__(self, limits=None, user_api=None):
        self._enabled = limits is not None
        if self._enabled:
            self.old_limits = get_threadpool_limits()
            _set_threadpool_limits(limits=limits, user_api=user_api)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.unregister()

    def unregister(self):
        if self._enabled:
            _set_threadpool_limits(limits=self.old_limits)
