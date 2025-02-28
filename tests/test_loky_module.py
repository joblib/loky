import multiprocessing as mp
import os
import sys
import shutil
import subprocess
import tempfile
import warnings
from subprocess import check_output

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

    # The following will always run using the Python 3.7 docker image.
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
