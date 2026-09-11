import multiprocessing as mp
import os
import sys
import shutil
import subprocess
import warnings
from subprocess import check_output
from unittest.mock import patch, mock_open

import pytest

import loky
from loky import cpu_count
from loky.backend.context import (
    _cpu_count_affinity_set,
    _MAX_WINDOWS_WORKERS,
)


def test_version():
    assert hasattr(
        loky, "__version__"
    ), "There are no __version__ argument on the loky module"


def test_cpu_count(monkeypatch):

    # Monkeypatch subprocess.run to simulate the absence of CIM on windows to
    # test the different code paths in _cpu_count_physical.
    old_run = subprocess.run

    def mock_run(*args, **kwargs):
        if (
            "powershell.exe" in args[0]
            and os.environ.get("LOKY_TEST_NO_CIM") == "true"
        ):
            raise RuntimeError("Cim not available")

        return old_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)

    cpus = cpu_count()
    assert type(cpus) is int
    assert cpus >= 1

    cpus_physical = cpu_count(only_physical_cores=True)
    assert type(cpus_physical) is int
    assert 1 <= cpus_physical <= cpus

    # again to check that it's correctly cached
    cpus_physical = cpu_count(only_physical_cores=True)
    assert type(cpus_physical) is int
    assert 1 <= cpus_physical <= cpus


@pytest.mark.skipif(sys.platform != "win32", reason="Windows specific test")
def test_windows_max_cpu_count():
    assert cpu_count() <= _MAX_WINDOWS_WORKERS


@pytest.mark.skipif(sys.platform != "win32", reason="Windows specific test")
@pytest.mark.parametrize(
    "implementation_name",
    [
        "_count_physical_cores_win32_ctypes",
        "_count_physical_cores_win32_powershell",
    ],
)
def test_windows_physical_cores(implementation_name):
    psutil = pytest.importorskip("psutil")
    implementation = getattr(loky.backend.context, implementation_name)

    expected = psutil.cpu_count(logical=False)
    assert expected is not None
    assert implementation() == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows specific test")
def test_windows_physical_cores_falls_back_to_powershell():
    from loky.backend.context import _count_physical_cores_win32

    with patch(
        "loky.backend.context._count_physical_cores_win32_ctypes",
        side_effect=RuntimeError("ctypes failed"),
    ):
        with patch(
            "loky.backend.context._count_physical_cores_win32_powershell",
            return_value=8,
        ) as mock_powershell:
            count = _count_physical_cores_win32()

    assert count == 8
    mock_powershell.assert_called_once()


def test_windows_physical_cores_powershell_sums_sockets_mine(monkeypatch):
    from loky.backend.context import _count_physical_cores_win32_powershell

    completed_process = subprocess.CompletedProcess([], 0, stdout="4\n4\n")
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: completed_process
    )

    assert _count_physical_cores_win32_powershell() == 8


cpu_count_cmd = (
    "from loky.backend.context import cpu_count;" "print(cpu_count({args}))"
)


def _patch_proc_cpuinfo(monkeypatch, content=None, error=None):
    # Monkeypatch open() so that reading /proc/cpuinfo returns `content` (or
    # raises `error`), without disturbing other files opened while computing
    # cpu_count() (e.g. Cgroup files).
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            if error is not None:
                raise error
            return mock_open(read_data=content)()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_cpu_count_os_sched_getaffinity():
    if not hasattr(os, "sched_getaffinity") or not hasattr(shutil, "which"):
        pytest.skip()

    taskset_bin = shutil.which("taskset")
    python_bin = shutil.which("python")

    if taskset_bin is None or python_bin is None:
        raise pytest.skip()

    try:
        os.sched_getaffinity(0)
    except NotImplementedError:
        pytest.skip()

    res = check_output(
        [
            taskset_bin,
            "-c",
            "0",
            python_bin,
            "-c",
            cpu_count_cmd.format(args=""),
        ],
        text=True,
    )

    res_physical = check_output(
        [
            taskset_bin,
            "-c",
            "0",
            python_bin,
            "-c",
            cpu_count_cmd.format(args="only_physical_cores=True"),
        ],
        text=True,
    )

    assert res.strip() == "1"
    assert res_physical.strip() == "1"


def test_cpu_count_psutil_affinity():
    psutil = pytest.importorskip("psutil")
    p = psutil.Process()
    if not hasattr(p, "cpu_affinity"):
        pytest.skip("psutil does not provide cpu_affinity on this platform")

    original_affinity = p.cpu_affinity()
    assert cpu_count() <= len(original_affinity)
    try:
        new_affinity = original_affinity[:1]
        p.cpu_affinity(new_affinity)
        assert cpu_count() == 1
    finally:
        p.cpu_affinity(original_affinity)


def test_cpu_count_cgroup_limit():
    if sys.platform == "win32":
        pytest.skip()

    if not hasattr(shutil, "which"):
        pytest.skip()

    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise pytest.skip("docker is required to run this test")

    loky_module_path = os.path.abspath(os.path.dirname(loky.__file__))
    loky_project_path = os.path.abspath(
        os.path.join(loky_module_path, os.pardir)
    )

    # Check if Docker can actually set cgroup CPU limits in this environment
    # by verifying that --cpus flag writes to cgroup files
    cgroup_check = check_output(
        f'{docker_bin} run --rm --cpus 0.5 python:3.10 python3 -c "'
        "import os; "
        "v2 = '/sys/fs/cgroup/cpu.max'; "
        "v1_quota = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'; "
        "v2_content = open(v2).read().strip() if os.path.exists(v2) else ''; "
        "v1_content = open(v1_quota).read().strip() if os.path.exists(v1_quota) else ''; "
        "print('ok' if (v2_content and v2_content != 'max') or (v1_content and v1_content != '-1') else 'skip')"
        '"',
        shell=True,
        text=True,
    ).strip()

    if cgroup_check != "ok":
        pytest.skip(
            "Docker doesn't properly set cgroup CPU limits in this environment"
        )

    # The following will always run using the Python 3.10 docker image.
    # We mount the loky source as /loky inside the container,
    # so it can be imported when running commands under /

    # Tell docker to configure the Cgroup quota to use 0.5 CPU, loky will
    # always detect 1 CPU because it rounds up to the next integer.
    res_500_mCPU = int(
        check_output(
            f"{docker_bin} run --rm --cpus 0.5 -v {loky_project_path}:/loky python:3.10 "
            f"/bin/bash -c 'pip install --quiet -e /loky ; "
            f"python -c \"{cpu_count_cmd.format(args='')}\"'",
            shell=True,
            text=True,
        ).strip()
    )
    assert res_500_mCPU == 1

    # Limiting to 1.5 CPUs can lead to 1 if there is only 1 CPU on the machine or
    # 2 if there are 2 CPUs or more.
    res_1500_mCPU = int(
        check_output(
            f"{docker_bin} run --rm --cpus 1.5 -v {loky_project_path}:/loky python:3.10 "
            f"/bin/bash -c 'pip install --quiet -e /loky ; "
            f"python -c \"{cpu_count_cmd.format(args='')}\"'",
            shell=True,
            text=True,
        ).strip()
    )
    assert res_1500_mCPU in (1, 2)

    # By default there is no limit: use all available CPUs.
    res_default = int(
        check_output(
            f"{docker_bin} run --rm -v {loky_project_path}:/loky python:3.10 "
            f"/bin/bash -c 'pip install --quiet -e /loky ; "
            f"python -c \"{cpu_count_cmd.format(args='')}\"'",
            shell=True,
            text=True,
        ).strip()
    )
    assert res_default >= res_1500_mCPU


def test_only_physical_cores_error(monkeypatch):
    # Check the warning issued by cpu_count(only_physical_cores=True) when
    # unable to retrieve the number of physical cores.
    if sys.platform != "linux":
        pytest.skip()

    # if number of available cpus is already restricted, cpu_count will return
    # that value and no warning is issued even if only_physical_cores == True.
    # (tested in another test: test_only_physical_cores_with_user_limitation
    cpu_count_mp = mp.cpu_count()
    if cpu_count() < cpu_count_mp:
        pytest.skip()

    # Simulate /proc/cpuinfo being unreadable so that the physical core
    # count cannot be found.
    _patch_proc_cpuinfo(
        monkeypatch, error=OSError("simulated /proc/cpuinfo read failure")
    )

    # clear the cache otherwise the warning is not triggered
    import loky.backend.context

    monkeypatch.setattr(loky.backend.context, "physical_cores_cache", {})

    with pytest.warns(
        UserWarning,
        match="Could not find the number of physical cores",
    ):
        # Falls back to the logical CPU count when the physical core count
        # cannot be found.
        assert cpu_count(only_physical_cores=True) == cpu_count_mp

    # Should not warn the second time
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cpu_count(only_physical_cores=True) == cpu_count_mp


def test_only_physical_cores_with_user_limitation():
    # Check that user limitation for the available number of cores is
    # respected even if only_physical_cores == True. On Linux, if the
    # restriction comes from CPU affinity and some of the affinity-pinned
    # logical CPUs are SMT siblings of the same physical core, the physical
    # core count can be strictly lower than the user limitation (see
    # test_cpu_count_only_physical_cores_smt_siblings_affinity below).
    cpu_count_mp = mp.cpu_count()
    cpu_count_user = cpu_count()

    if cpu_count_user < cpu_count_mp:
        cpu_affinity_set = _cpu_count_affinity_set()
        affinity_cpu_count = (
            cpu_count_mp if cpu_affinity_set is None else len(cpu_affinity_set)
        )
        if affinity_cpu_count < cpu_count_mp:
            # The restriction includes a CPU affinity component: the SMT
            # collapsing logic may legitimately kick in and report fewer
            # physical cores than cpu_count_user.
            assert cpu_count(only_physical_cores=True) <= cpu_count_user
        else:
            # The restriction only comes from Cgroup/LOKY_MAX_CPU_COUNT:
            # only_physical_cores must not be enforced, see cpu_count's
            # docstring.
            assert cpu_count(only_physical_cores=True) == cpu_count_user


def test_cpu_count_only_physical_cores_smt_siblings_affinity(monkeypatch):
    # Regression test for https://github.com/joblib/loky/issues/639:
    # only_physical_cores=True should collapse SMT/hyper-threading sibling
    # logical CPUs sharing the same physical core when the usable CPUs are
    # restricted through CPU affinity (e.g. `taskset`), instead of just
    # returning the (affinity-restricted) logical CPU count unchanged.
    if sys.platform != "linux":
        pytest.skip("Linux specific test")

    import loky.backend.context as context

    # Simulate a 4 logical CPU machine: CPU 0 and 1 are SMT siblings of
    # physical core 0; CPU 2 and 3 are SMT siblings of physical core 1.
    fake_cpuinfo = "".join(
        f"processor\t: {cpu}\nphysical id\t: 0\ncore id\t: {cpu // 2}\n\n"
        for cpu in range(4)
    )

    _patch_proc_cpuinfo(monkeypatch, content=fake_cpuinfo)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    monkeypatch.setattr(context, "physical_cores_cache", {})

    # taskset -c 0,1 pins the process to 2 logical CPUs that are SMT
    # siblings of a single physical core.
    assert context.cpu_count() == 2
    assert context.cpu_count(only_physical_cores=True) == 1

    # Changing the affinity to 2 logical CPUs that belong to different
    # physical cores (0 and 2) must not collapse them, and must use a
    # separate cache entry than the previous affinity set.
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 2}, raising=False
    )
    assert context.cpu_count() == 2
    assert context.cpu_count(only_physical_cores=True) == 2

    # Lifting the affinity restriction entirely must report the 2 physical
    # cores of the whole machine, going through the un-keyed cache entry.
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False
    )
    assert context.cpu_count() == 4
    assert context.cpu_count(only_physical_cores=True) == 2


def test_count_physical_cores_linux_multi_socket(monkeypatch):
    # Regression test: a physical core must be identified by its
    # (physical id, core id) pair, not by core id alone, otherwise cores
    # sharing the same core id across sockets on a multi-socket machine
    # get under-counted.
    if sys.platform != "linux":
        pytest.skip("Linux specific test")

    from loky.backend.context import _count_physical_cores_linux

    # Simulate a 2-socket machine with 2 physical cores per socket, where
    # each socket reuses core id 0 and 1.
    fake_cpuinfo = "".join(
        f"processor\t: {cpu}\nphysical id\t: {cpu // 2}\ncore id\t: {cpu % 2}\n\n"
        for cpu in range(4)
    )

    _patch_proc_cpuinfo(monkeypatch, content=fake_cpuinfo)

    assert _count_physical_cores_linux() == 4


def test_cpu_count_os_sched_getaffinity_smt_siblings():
    # End-to-end version of the test above: actually pin the current
    # process to 2 SMT sibling logical CPUs of the same physical core (if
    # such a pair can be found on the machine running the test) and check
    # that only_physical_cores=True correctly reports 1 physical core while
    # the plain logical CPU count reports 2.
    if sys.platform != "linux":
        pytest.skip("Linux specific test")

    if not hasattr(os, "sched_getaffinity"):
        pytest.skip()

    psutil = pytest.importorskip("psutil")
    p = psutil.Process()
    if not hasattr(p, "cpu_affinity"):
        pytest.skip("psutil does not provide cpu_affinity on this platform")

    def _expand(cpu_range):
        if "-" in cpu_range:
            start, end = cpu_range.split("-")
            return list(range(int(start), int(end) + 1))
        return [int(cpu_range)]

    smt_pair = None
    for cpu in range(os.cpu_count() or 0):
        siblings_path = (
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        try:
            with open(siblings_path) as f:
                content = f.read().strip()
        except OSError:
            pytest.skip("CPU topology information not available")

        siblings = sorted(
            {c for part in content.split(",") for c in _expand(part)}
        )
        if len(siblings) >= 2:
            smt_pair = siblings[:2]
            break

    if smt_pair is None:
        pytest.skip(
            "could not find 2 SMT sibling logical CPUs on this machine"
        )

    original_affinity = p.cpu_affinity()
    try:
        p.cpu_affinity(smt_pair)
        assert cpu_count() == 2
        assert cpu_count(only_physical_cores=True) == 1
    finally:
        p.cpu_affinity(original_affinity)


@pytest.mark.parametrize(
    "read_data,description",
    [
        ("", "empty file"),
        ("max\n", "max value"),
    ],
)
def test_cpu_count_cgroup_invalid_content(read_data, description):
    # Test that invalid cgroup cpu.max file content is handled gracefully
    # and doesn't cause a ValueError when trying to unpack values
    if sys.platform != "linux":
        pytest.skip()

    from loky.backend.context import _cpu_count_cgroup

    os_cpu_count = mp.cpu_count()

    # Mock the file with the provided read_data
    with patch("builtins.open", mock_open(read_data=read_data)):
        with patch("os.path.exists") as mock_exists:
            # cpu.max exists, but other files don't
            mock_exists.side_effect = (
                lambda path: path == "/sys/fs/cgroup/cpu.max"
            )

            # This should not raise ValueError and return os_cpu_count
            result = _cpu_count_cgroup(os_cpu_count)
            assert (
                result == os_cpu_count
            ), f"cpu.max with {description} should return os_cpu_count"
