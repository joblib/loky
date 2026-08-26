import multiprocessing as mp
import os
import sys
import shutil
import subprocess
import tempfile
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

    # Monkeypatch subprocess.run to simulate the absence of lscpu on linux or CIM on
    # windows to test the different code paths in _cpu_count_physical.
    old_run = subprocess.run

    def mock_run(*args, **kwargs):
        if (
            "lscpu" in args[0]
            and os.environ.get("LOKY_TEST_NO_LSCPU") == "true"
        ):
            raise RuntimeError("lscpu not available")

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


cpu_count_cmd = (
    "from loky.backend.context import cpu_count;" "print(cpu_count({args}))"
)


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


def test_only_physical_cores_error():
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write bad lscpu and cat programs so that both the primary lscpu
        # based method and the /proc/cpuinfo fallback fail to find the
        # number of physical cores.
        for cmd in ["lscpu", "cat"]:
            cmd_path = f"{tmp_dir}/{cmd}"
            with open(cmd_path, "w") as f:
                f.write("#!/bin/sh\n" "exit(1)")
            os.chmod(cmd_path, 0o777)

        try:
            old_path = os.environ["PATH"]
            os.environ["PATH"] = f"{tmp_dir}:{old_path}"

            # clear the cache otherwise the warning is not triggered
            import loky.backend.context

            loky.backend.context.physical_cores_cache = None

            with pytest.warns(
                UserWarning,
                match="Could not find the number of" " physical cores",
            ):
                cpu_count(only_physical_cores=True)

            # Should not warn the second time
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                cpu_count(only_physical_cores=True)

        finally:
            os.environ["PATH"] = old_path


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
    fake_lscpu_stdout = "0,0,0\n1,0,0\n2,1,0\n3,1,0\n"

    old_run = subprocess.run

    def mock_run(args, **kwargs):
        if isinstance(args, list) and args[:1] == ["lscpu"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=fake_lscpu_stdout, stderr=""
            )
        return old_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    monkeypatch.setattr(context, "physical_cores_affinity_cache", {})

    # taskset -c 0,1 pins the process to 2 logical CPUs that are SMT
    # siblings of a single physical core.
    assert context.cpu_count() == 2
    assert context.cpu_count(only_physical_cores=True) == 1


def test_cpu_count_os_sched_getaffinity_smt_siblings():
    # End-to-end version of the test above: actually use taskset to pin the
    # current process to 2 SMT sibling logical CPUs of the same physical
    # core (if such a pair can be found on the machine running the test)
    # and check that only_physical_cores=True correctly reports 1 physical
    # core while the plain logical CPU count reports 2.
    if sys.platform != "linux":
        pytest.skip("Linux specific test")

    if not hasattr(os, "sched_getaffinity") or not hasattr(shutil, "which"):
        pytest.skip()

    taskset_bin = shutil.which("taskset")
    python_bin = shutil.which("python")
    if taskset_bin is None or python_bin is None:
        pytest.skip()

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

    cpu_list = ",".join(str(c) for c in smt_pair)

    res = check_output(
        [
            taskset_bin,
            "-c",
            cpu_list,
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
            cpu_list,
            python_bin,
            "-c",
            cpu_count_cmd.format(args="only_physical_cores=True"),
        ],
        text=True,
    )

    assert res.strip() == "2"
    assert res_physical.strip() == "1"


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
