###############################################################################
# Basic context management with LokyContext
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/context.py
#  * Create a context ensuring loky uses only objects that are compatible
#  * Add LokyContext to the list of context of multiprocessing so loky can be
#    used with multiprocessing.set_start_method
#  * Implement a CFS-aware amd physical-core aware cpu_count function.
#
import ctypes
import math
import multiprocessing as mp
import os
import posixpath
import re
import subprocess
import sys
import traceback
import warnings

from ctypes import wintypes
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from concurrent.futures.process import _MAX_WINDOWS_WORKERS


from .process import LokyProcess, LokyInitMainProcess

# Apparently, on older Python versions, loky cannot work 61 workers on Windows
# but instead 60: ¯\_(ツ)_/¯
if sys.version_info < (3, 10):
    _MAX_WINDOWS_WORKERS = _MAX_WINDOWS_WORKERS - 1

START_METHODS = ["loky", "loky_init_main", "spawn"]
if sys.platform != "win32":
    START_METHODS += ["fork", "forkserver"]

_DEFAULT_START_METHOD = None

# Cache for the number of physical cores, to avoid repeating subprocess
# calls or re-parsing /proc/cpuinfo. Keyed by None for the whole machine, or
# by a frozenset of logical CPU ids when restricted to a CPU affinity mask,
# see `_count_physical_cores`.
physical_cores_cache = {}


def get_context(method=None):
    # Try to overload the default context
    method = method or _DEFAULT_START_METHOD or "loky"
    if method == "fork":
        # If 'fork' is explicitly requested, warn user about potential issues.
        warnings.warn(
            "`fork` start method should not be used with "
            "`loky` as it does not respect POSIX. Try using "
            "`spawn` or `loky` instead.",
            UserWarning,
        )
    try:
        return mp_get_context(method)
    except ValueError:
        raise ValueError(
            f"Unknown context '{method}'. Value should be in "
            f"{START_METHODS}."
        )


def set_start_method(method, force=False):
    global _DEFAULT_START_METHOD
    if _DEFAULT_START_METHOD is not None and not force:
        raise RuntimeError("context has already been set")
    assert method is None or method in START_METHODS, (
        f"'{method}' is not a valid start_method. It should be in "
        f"{START_METHODS}"
    )

    _DEFAULT_START_METHOD = method


def get_start_method():
    return _DEFAULT_START_METHOD


def cpu_count(only_physical_cores=False):
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available on some Unix systems);
     * Cgroup CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.

    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT): this is
    given as the minimum of the same constraints as above, together with the
    number of physical cores.
    On Linux, the computed number of physical cores is restrained to the
    physical cores reachable through the CPU affinity mask.
    On other platforms, it's the number of physical cores on the whole
    machine, since the affinity mask cannot be taken into account there.

    Note that on Windows, the returned number of CPUs cannot exceed 61 (or 60 for
    Python < 3.10), see:
    https://bugs.python.org/issue26903.

    It is also always larger or equal to 1.
    """
    # Note: os.cpu_count() is allowed to return None in its docstring
    os_cpu_count = os.cpu_count() or 1
    if sys.platform == "win32":
        # On Windows, attempting to use more than 61 CPUs would result in a
        # OS-level error. See https://bugs.python.org/issue26903. According to
        # https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
        # it might be possible to go beyond with a lot of extra work but this
        # does not look easy.
        os_cpu_count = min(os_cpu_count, _MAX_WINDOWS_WORKERS)

    cpu_affinity_set = _cpu_count_affinity_set()
    cpu_count_affinity = (
        os_cpu_count if cpu_affinity_set is None else len(cpu_affinity_set)
    )
    cpu_count_cgroup = _cpu_count_cgroup(os_cpu_count)
    cpu_count_loky = int(os.environ.get("LOKY_MAX_CPU_COUNT", os_cpu_count))

    if not only_physical_cores:
        return max(
            1,
            min(
                os_cpu_count,
                cpu_count_affinity,
                cpu_count_cgroup,
                cpu_count_loky,
            ),
        )

    cpu_count_physical, exception = _count_physical_cores(
        # On Linux, try to collapse SMT/hyper-threading sibling logical CPUs
        # reachable through the current CPU affinity mask that share the same
        # physical core. See https://github.com/joblib/loky/issues/639.
        # On other platforms we lack easy access to CPU topology info to refine
        # the count, so just pass None for `cpu_affinity_set`.
        cpu_affinity_set
        if sys.platform == "linux"
        else None
    )
    if cpu_count_physical == "not found":
        cpu_count_physical = os_cpu_count

    _warn_physical_cores_not_found(exception)

    return max(
        1,
        min(
            os_cpu_count,
            cpu_count_physical,
            cpu_count_affinity,
            cpu_count_cgroup,
            cpu_count_loky,
        ),
    )


def _warn_physical_cores_not_found(exception):
    if exception is not None:
        # warns only the first time
        warnings.warn(
            "Could not find the number of physical cores for the "
            f"following reason:\n{exception}\n"
            "Returning the number of logical cores instead. You can "
            "silence this warning by setting LOKY_MAX_CPU_COUNT to "
            "the number of cores you want to use."
        )
        traceback.print_tb(exception.__traceback__)


def _cgroup_v2_cpu_max_paths():
    """Find the current cgroup and its ancestors within visible v2 mounts."""
    fallback = ["/sys/fs/cgroup/cpu.max"]
    if not os.path.exists("/proc/self/cgroup"):
        return fallback
    try:
        with open(
            "/proc/self/cgroup",
            encoding=sys.getfilesystemencoding(),
            errors="surrogateescape",
            newline="\n",
        ) as fh:
            entries = [
                line.split(":", 2)
                for line in fh.read().removesuffix("\n").split("\n")
            ]
        # Do not trust a truncated path if membership has continuation lines
        # or more than one unified-hierarchy record.
        if any(
            len(entry) != 3
            or not entry[0].isdigit()
            or not entry[2].startswith("/")
            for entry in entries
        ):
            return fallback
        cgroups = [entry[2] for entry in entries if entry[:2] == ["0", ""]]
        if len(cgroups) != 1:
            return fallback
        cgroup = cgroups[0]
        # A cgroup outside the namespace can be reported as /../... . Do not
        # resolve that path outside the visible mount.
        if ".." in cgroup.split("/"):
            return fallback
        with open(
            "/proc/self/mountinfo",
            encoding=sys.getfilesystemencoding(),
            errors="surrogateescape",
            newline="\n",
        ) as fh:
            mounts = fh.readlines()
    except OSError:
        return fallback

    mount_by_id = {}
    for mount in mounts:
        fields, separator, filesystem = mount.partition(" - ")
        fields = fields.split(" ")
        if not separator or len(fields) < 6:
            continue
        # mountinfo octal-escapes space, tab, newline and backslash in paths.
        root, mountpoint = (
            re.sub(
                r"\\(040|011|012|134)",
                lambda match: chr(int(match[1], 8)),
                path,
            )
            for path in fields[3:5]
        )
        if not all(path.startswith("/") for path in (root, mountpoint)):
            continue
        if any(".." in path.split("/") for path in (root, mountpoint)):
            continue
        mount_by_id[fields[0]] = (
            fields[1],
            root.rstrip("/"),
            mountpoint.rstrip("/") or "/",
            filesystem.startswith("cgroup2 "),
        )

    paths = []
    matched_mount = False
    for mount_id, (
        parent_id,
        root,
        mountpoint,
        is_cgroup2,
    ) in mount_by_id.items():
        if not is_cgroup2:
            continue
        if cgroup != root and not cgroup.startswith(root + "/"):
            continue
        matched_mount = True
        ancestors = set()
        ancestor = mount_id
        while ancestor in mount_by_id and ancestor not in ancestors:
            ancestors.add(ancestor)
            ancestor = mount_by_id[ancestor][0]
        current = posixpath.join(mountpoint, cgroup[len(root) :].lstrip("/"))
        current = current.rstrip("/") or "/"
        while True:
            cpu_max = posixpath.join(current, "cpu.max")
            # Covered mounts remain in mountinfo. A different mount on this
            # path can expose an unrelated cgroup, or even another filesystem.
            if not any(
                other_id not in ancestors
                and (
                    cpu_max == other[2]
                    or cpu_max.startswith(other[2].rstrip("/") + "/")
                )
                for other_id, other in mount_by_id.items()
            ):
                paths.append(cpu_max)
            if current == mountpoint:
                break
            current = posixpath.dirname(current)
    return paths if matched_mount else fallback


def _cpu_count_cgroup(os_cpu_count):
    # cgroup v2 quotas can be attached to the process's cgroup or any parent.
    # https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
    found_v2 = False
    cpu_counts = []
    for cpu_max_fname in _cgroup_v2_cpu_max_paths():
        if not os.path.exists(cpu_max_fname):
            continue
        try:
            with open(cpu_max_fname) as fh:
                parts = fh.read().strip().split()
        except OSError:
            # The CPU controller may be absent, or the cgroup inaccessible.
            continue
        if len(parts) != 2:
            continue
        found_v2 = True
        cpu_quota_us, cpu_period_us = parts
        if cpu_quota_us != "max":
            cpu_quota_us = int(cpu_quota_us)
            cpu_period_us = int(cpu_period_us)
            if cpu_quota_us > 0 and cpu_period_us > 0:
                cpu_counts.append(math.ceil(cpu_quota_us / cpu_period_us))
    if found_v2:
        return min(cpu_counts) if cpu_counts else os_cpu_count

    # If we didn't get values from cgroup v2, try cgroup v1.
    # https://www.kernel.org/doc/html/latest/scheduler/sched-bwc.html#management
    cfs_quota_fname = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    cfs_period_fname = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(cfs_quota_fname) and os.path.exists(cfs_period_fname):
        with open(cfs_quota_fname) as fh:
            cpu_quota_us = fh.read().strip()
        with open(cfs_period_fname) as fh:
            cpu_period_us = fh.read().strip()
    else:
        cpu_quota_us = "max"

    if cpu_quota_us == "max":
        return os_cpu_count
    cpu_quota_us = int(cpu_quota_us)
    cpu_period_us = int(cpu_period_us)
    if cpu_quota_us > 0 and cpu_period_us > 0:
        return math.ceil(cpu_quota_us / cpu_period_us)
    # A negative quota disables cgroup CPU bandwidth limits.
    return os_cpu_count


def _cpu_count_affinity_set():
    """Return the current CPU affinity mask as a set of logical CPU ids.

    Return None if the affinity mask cannot be determined on this platform,
    for instance because neither `os.sched_getaffinity` nor `psutil` are
    available.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return os.sched_getaffinity(0)
        except NotImplementedError:
            pass

    # On some platforms, os.sched_getaffinity does not exist or raises
    # NotImplementedError, let's try with the psutil if installed.
    try:
        import psutil

        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            return set(p.cpu_affinity())

    except ImportError:  # pragma: no cover
        if (
            sys.platform == "linux"
            and os.environ.get("LOKY_MAX_CPU_COUNT") is None
        ):
            # Some platforms don't implement os.sched_getaffinity on Linux which
            # can cause severe oversubscription problems. Better warn the
            # user in this particularly pathological case which can wreck
            # havoc, typically on CI workers.
            warnings.warn(
                "Failed to inspect CPU affinity constraints on this system. "
                "Please install psutil or explictly set LOKY_MAX_CPU_COUNT."
            )

    return None


def _count_physical_cores(cpu_set=None):
    """Return a tuple (number of physical cores, exception)

    If the number of physical cores is found, exception is set to None.
    If it has not been found, return ("not found", exception).

    If `cpu_set` is not None, only the logical CPUs it contains are
    considered when counting physical cores, which also collapses
    SMT/hyper-threading sibling logical CPUs that share the same physical
    core. Only implemented for Linux, where per-CPU topology information is
    readily available; callers are expected to only pass a non-None
    `cpu_set` on Linux.

    The number of physical cores is cached (per distinct `cpu_set` when one
    is passed) to avoid repeating subprocess calls or re-parsing
    /proc/cpuinfo.
    """
    exception = None

    # First check if the value is cached
    global physical_cores_cache
    cache_key = None if cpu_set is None else frozenset(cpu_set)
    if cache_key in physical_cores_cache:
        return physical_cores_cache[cache_key], None

    # Not cached yet, find it
    try:
        if sys.platform == "linux":
            cpu_count_physical = _count_physical_cores_linux(cpu_set)
        elif cpu_set is not None:
            raise NotImplementedError(
                "affinity-aware physical core counting is only "
                "implemented on Linux"
            )
        elif sys.platform == "win32":
            cpu_count_physical = _count_physical_cores_win32()
        elif sys.platform == "darwin":
            cpu_count_physical = _count_physical_cores_darwin()
        elif sys.platform.startswith("freebsd"):
            cpu_count_physical = _count_physical_cores_freebsd()
        else:
            raise NotImplementedError(f"unsupported platform: {sys.platform}")

        # if cpu_count_physical < 1, we did not find a valid value
        if cpu_count_physical < 1:
            raise ValueError(f"found {cpu_count_physical} physical cores < 1")

    except Exception as e:
        exception = e
        cpu_count_physical = "not found"

    # Put the result in cache
    physical_cores_cache[cache_key] = cpu_count_physical

    return cpu_count_physical, exception


def _count_physical_cores_linux(cpu_set=None):
    """Return the number of distinct physical cores on Linux, by parsing
    /proc/cpuinfo.

    A physical core is identified by its (physical id, core id) pair: core
    id alone is only guaranteed unique within a physical package, so a
    multi-socket machine can otherwise under-count cores that share the
    same core id across sockets.

    If `cpu_set` is not None, only the logical CPUs it contains are
    considered, which also collapses SMT/hyper-threading sibling logical
    CPUs that share the same physical core.
    """
    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()

    cores = set()
    processor = core_id = physical_id = None
    for line in cpuinfo.splitlines() + [""]:
        if not line.strip():
            # blank line: end of the current logical CPU block
            if core_id is not None and (
                cpu_set is None or processor in cpu_set
            ):
                cores.add((physical_id, core_id))
            processor = core_id = physical_id = None
            continue

        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key == "processor":
            processor = int(value)
        elif key == "core id":
            core_id = value
        elif key == "physical id":
            physical_id = value

    if not cores:
        raise ValueError("could not find any physical core in /proc/cpuinfo")

    return len(cores)


def _count_physical_cores_win32():
    try:
        return _count_physical_cores_win32_ctypes()
    except Exception:
        pass  # fallback to powershell
    try:
        return _count_physical_cores_win32_powershell()
    except Exception:
        pass  # fallback to wmic (older Windows versions; deprecated now)

    cpu_info = subprocess.run(
        "wmic CPU Get NumberOfCores /Format:csv".split(),
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    cpu_info = cpu_info.stdout.splitlines()
    cpu_info = [
        l.split(",")[1] for l in cpu_info if (l and l != "Node,NumberOfCores")
    ]
    return sum(map(int, cpu_info))


def _count_physical_cores_win32_powershell():
    cmd = "-NoProfile -Command (Get-CimInstance -ClassName Win32_Processor).NumberOfCores"
    cpu_info = subprocess.run(
        f"powershell.exe {cmd}".split(),
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    cpu_info = cpu_info.stdout.splitlines()
    return sum(map(int, cpu_info))


def _count_physical_cores_win32_ctypes():
    ERROR_INSUFFICIENT_BUFFER = 122
    RelationProcessorCore = 0

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_logical_processor_information = (
        kernel32.GetLogicalProcessorInformationEx
    )
    get_logical_processor_information.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_logical_processor_information.restype = wintypes.BOOL

    # Mirror the header of SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX. The full
    # structure is variable-sized, and only Relationship and Size are needed.
    # https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_logical_processor_information_ex
    class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
        _fields_ = [
            ("Relationship", wintypes.DWORD),
            ("Size", wintypes.DWORD),
        ]

    # First obtain the required buffer size. This call is expected to fail with
    # ERROR_INSUFFICIENT_BUFFER and set returned_length.
    # https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex
    returned_length = wintypes.DWORD()
    returned_length_ref = ctypes.byref(returned_length)
    if get_logical_processor_information(
        RelationProcessorCore, None, returned_length_ref
    ):
        raise RuntimeError("unexpected successful buffer sizing call")

    error = ctypes.get_last_error()
    if error != ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(error)

    buf = ctypes.create_string_buffer(returned_length.value)
    if not get_logical_processor_information(
        RelationProcessorCore, buf, returned_length_ref
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    offset = 0
    physical_core_count = 0
    header_size = ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)

    while offset < returned_length.value:
        remaining = returned_length.value - offset
        if remaining < header_size:
            raise RuntimeError("truncated processor information record")

        processor_core_info = (
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.from_buffer(buf, offset)
        )
        record_size = processor_core_info.Size
        if record_size < header_size or record_size > remaining:
            raise RuntimeError("invalid processor information record size")
        if processor_core_info.Relationship != RelationProcessorCore:
            raise RuntimeError("unexpected logical processor relationship")

        physical_core_count += 1
        offset += record_size

    if physical_core_count == 0:
        raise RuntimeError("Windows reported no active physical cores")

    if physical_core_count < 1:
        raise RuntimeError(
            "GetLogicalProcessorInformationEx returned no physical cores"
        )
    return physical_core_count


def _count_physical_cores_darwin():
    cpu_info = subprocess.run(
        "sysctl -n hw.physicalcpu".split(),
        capture_output=True,
        text=True,
    )
    cpu_info = cpu_info.stdout
    return int(cpu_info)


def _count_physical_cores_freebsd():
    cpu_info = subprocess.run(
        "sysctl -n kern.smp.cores".split(),
        capture_output=True,
        text=True,
    )
    cpu_info = cpu_info.stdout
    return int(cpu_info)


class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""

    _name = "loky"
    Process = LokyProcess
    cpu_count = staticmethod(cpu_count)

    def Queue(self, maxsize=0, reducers=None):
        """Returns a queue object"""
        from .queues import Queue

        return Queue(maxsize, reducers=reducers, ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        """Returns a queue object"""
        from .queues import SimpleQueue

        return SimpleQueue(reducers=reducers, ctx=self.get_context())

    if sys.platform != "win32":
        """For Unix platform, use our custom implementation of synchronize
        ensuring that we use the loky.backend.resource_tracker to clean-up
        the semaphores in case of a worker crash.
        """

        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from .synchronize import Semaphore

            return Semaphore(value=value)

        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore

            return BoundedSemaphore(value)

        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock

            return Lock()

        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock

            return RLock()

        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition

            return Condition(lock)

        def Event(self):
            """Returns an event object"""
            from .synchronize import Event

            return Event()


class LokyInitMainContext(LokyContext):
    """Extra context with LokyProcess, which does load the main module

    This context is used for compatibility in the case ``cloudpickle`` is not
    present on the running system. This permits to load functions defined in
    the ``main`` module, using proper safeguards. The declaration of the
    ``executor`` should be protected by ``if __name__ == "__main__":`` and the
    functions and variable used from main should be out of this block.

    This mimics the default behavior of multiprocessing under Windows and the
    behavior of the ``spawn`` start method on a posix system.
    For more details, see the end of the following section of python doc
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """

    _name = "loky_init_main"
    Process = LokyInitMainProcess


# Register loky context so it works with multiprocessing.get_context
ctx_loky = LokyContext()
mp.context._concrete_contexts["loky"] = ctx_loky
mp.context._concrete_contexts["loky_init_main"] = LokyInitMainContext()
