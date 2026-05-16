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
from loky.backend.context import _cpu_count_user, _MAX_WINDOWS_WORKERS


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
    if _cpu_count_user(cpu_count_mp) < cpu_count_mp:
        pytest.skip()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write bad lscpu program
        lscpu_path = f"{tmp_dir}/lscpu"
        with open(lscpu_path, "w") as f:
            f.write("#!/bin/sh\n" "exit(1)")
        os.chmod(lscpu_path, 0o777)

        try:
            old_path = os.environ["PATH"]
            os.environ["PATH"] = f"{tmp_dir}:{old_path}"

            # clear the cache otherwise the warning is not triggered
            import loky.backend.context

            loky.backend.context.physical_cores_cache = None

            with (
                patch.object(
                    loky.backend.context,
                    "_count_performance_cores_linux",
                    return_value=None,
                ),
                patch.object(
                    loky.backend.context,
                    "_count_physical_cores_linux",
                    side_effect=RuntimeError("physical core probe failed"),
                ),
            ):
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
    # respected even if only_physical_cores == True
    cpu_count_mp = mp.cpu_count()
    cpu_count_user = _cpu_count_user(cpu_count_mp)

    if cpu_count_user < cpu_count_mp:
        assert cpu_count() == cpu_count_user
        assert cpu_count(only_physical_cores=True) == cpu_count_user


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


def test_count_physical_cores_prefers_performance_cores(monkeypatch):
    import loky.backend.context as context

    monkeypatch.setattr(context.sys, "platform", "linux")
    monkeypatch.setattr(
        context,
        "_count_performance_cores_linux",
        lambda: 4,
    )
    monkeypatch.setattr(
        context,
        "_count_physical_cores_linux",
        lambda: 8,
    )
    monkeypatch.setattr(context, "physical_cores_cache", None)

    first_value, first_exc = context._count_physical_cores()
    second_value, second_exc = context._count_physical_cores()

    assert isinstance(context.physical_cores_cache, dict)
    assert context.physical_cores_cache["preferred"] == 4
    assert first_value == 4
    assert first_exc is None
    assert second_value == 4
    assert second_exc is None


def test_count_physical_cores_fallback_to_physical_on_ambiguity(monkeypatch):
    import loky.backend.context as context

    monkeypatch.setattr(context.sys, "platform", "linux")
    monkeypatch.setattr(
        context,
        "_count_performance_cores_linux",
        lambda: None,
    )
    monkeypatch.setattr(
        context,
        "_count_physical_cores_linux",
        lambda: 8,
    )
    monkeypatch.setattr(context, "physical_cores_cache", None)

    value, exc = context._count_physical_cores()

    assert value == 8
    assert exc is None


def test_count_performance_cores_linux_missing_sysfs(monkeypatch):
    import loky.backend.context as context
    import glob

    monkeypatch.setattr(glob, "glob", lambda _: [])
    assert context._count_performance_cores_linux() is None


def test_count_performance_cores_linux_no_hybrid_frequency_data(
    monkeypatch,
):
    import loky.backend.context as context
    import glob

    cpu_paths = [
        "/sys/devices/system/cpu/cpu0",
        "/sys/devices/system/cpu/cpu1",
    ]
    monkeypatch.setattr(glob, "glob", lambda _: cpu_paths)

    file_contents = {
        # Identical base frequencies across all CPUs do not indicate
        # a hybrid topology.
        "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency": "1900000",
        "/sys/devices/system/cpu/cpu1/cpufreq/base_frequency": "1900000",
    }

    def _open_side_effect(path, *args, **kwargs):
        normalized_path = path.replace("\\", "/")
        if normalized_path in file_contents:
            return mock_open(read_data=file_contents[normalized_path])()
        raise OSError()

    with patch("builtins.open", side_effect=_open_side_effect):
        assert context._count_performance_cores_linux() is None


def test_count_performance_cores_win32_ctypes_failure(monkeypatch):
    import loky.backend.context as context

    from types import SimpleNamespace

    def _raise_os_error(*args, **kwargs):
        raise OSError()

    monkeypatch.setitem(
        context.sys.modules,
        "ctypes",
        SimpleNamespace(WinDLL=_raise_os_error),
    )
    assert context._count_performance_cores_win32() is None


def test_count_performance_cores_darwin_missing_perflevel(monkeypatch):
    import loky.backend.context as context

    def _mock_run(cmd, capture_output, text):
        class _Result:
            def __init__(self, stdout):
                self.stdout = stdout

        if cmd == ["sysctl", "-n", "hw.nperflevels"]:
            return _Result("2\n")
        if cmd == ["sysctl", "-n", "hw.perflevel0.physicalcpu"]:
            return _Result("")
        raise RuntimeError("unexpected command")

    monkeypatch.setattr(context.subprocess, "run", _mock_run)
    assert context._count_performance_cores_darwin() is None


def test_count_performance_cores_linux_hybrid_data(monkeypatch):
    import loky.backend.context as context
    import glob

    cpu_paths = [
        "/sys/devices/system/cpu/cpu0",
        "/sys/devices/system/cpu/cpu1",
    ]
    monkeypatch.setattr(glob, "glob", lambda _: cpu_paths)

    file_contents = {
        # Two cores with different base frequencies: one performance core and
        # one efficiency core.
        "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency": "1900000",
        "/sys/devices/system/cpu/cpu1/cpufreq/base_frequency": "1600000",
    }

    def _open_side_effect(path, *args, **kwargs):
        normalized_path = path.replace("\\", "/")
        if normalized_path in file_contents:
            return mock_open(read_data=file_contents[normalized_path])()
        raise OSError()

    with patch("builtins.open", side_effect=_open_side_effect):
        assert context._count_performance_cores_linux() == 1


def test_cpu_count_only_performance_cores_false_disables_filtering(
    monkeypatch,
):
    import loky.backend.context as context

    monkeypatch.setattr(context.sys, "platform", "linux")
    monkeypatch.setattr(context.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(context, "_cpu_count_user", lambda _: 8)
    monkeypatch.setattr(context, "_count_preferred_cores_linux", lambda: 4)
    monkeypatch.setattr(context, "_count_physical_cores_linux", lambda: 8)
    monkeypatch.setattr(context, "physical_cores_cache", {})

    assert context.cpu_count(only_physical_cores=True) == 4
    assert (
        context.cpu_count(
            only_physical_cores=True, only_performance_cores=False
        )
        == 8
    )


def test_cpu_count_only_performance_cores_invalid_value():
    with pytest.raises(ValueError, match="only_performance_cores"):
        cpu_count(only_physical_cores=True, only_performance_cores="wrong")


def test_cpu_count_only_performance_cores_ignored_without_physical_cores(
    monkeypatch,
):
    import loky.backend.context as context

    monkeypatch.setattr(context.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(context, "_cpu_count_user", lambda _: 8)

    # Invalid values are ignored when only_physical_cores is False.
    assert context.cpu_count(only_performance_cores="wrong") == 8


def test_count_physical_cores_cache_is_split_by_mode(monkeypatch):
    import loky.backend.context as context

    monkeypatch.setattr(context.sys, "platform", "linux")
    monkeypatch.setattr(context, "_count_preferred_cores_linux", lambda: 4)
    monkeypatch.setattr(context, "_count_physical_cores_linux", lambda: 8)
    monkeypatch.setattr(context, "physical_cores_cache", {})

    preferred_value, preferred_exc = context._count_physical_cores(
        only_performance_cores=True
    )
    physical_value, physical_exc = context._count_physical_cores(
        only_performance_cores=False
    )

    assert preferred_value == 4
    assert preferred_exc is None
    assert physical_value == 8
    assert physical_exc is None
