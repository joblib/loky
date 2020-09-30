import multiprocessing as mp
import os
import sys
import shutil
import tempfile
from subprocess import check_output
import subprocess

import pytest

import loky
from loky import cpu_count
from loky.backend.context import _cpu_count_user


def test_version():
    assert hasattr(loky, '__version__'), (
        "There are no __version__ argument on the loky module")


def test_cpu_count():
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


cpu_count_cmd = ("from loky.backend.context import cpu_count;"
                 "print(cpu_count({args}))")


def test_cpu_count_affinity():
    if not hasattr(os, 'sched_getaffinity') or not hasattr(shutil, 'which'):
        pytest.skip()

    taskset_bin = shutil.which('taskset')
    python_bin = shutil.which('python')

    if taskset_bin is None or python_bin is None:
        raise pytest.skip()

    try:
        os.sched_getaffinity(0)
    except NotImplementedError:
        pytest.skip()

    res = check_output([taskset_bin, '-c', '0',
                        python_bin, '-c', cpu_count_cmd.format(args='')])
    
    res_physical = check_output([
        taskset_bin, '-c', '0', python_bin, '-c',
        cpu_count_cmd.format(args='only_physical_cores=True')])

    assert res.strip().decode('utf-8') == '1'
    assert res_physical.strip().decode('utf-8') == '1'


def test_cpu_count_cfs_limit():
    if sys.platform == "win32":
        pytest.skip()

    if not hasattr(shutil, 'which'):
        pytest.skip()

    docker_bin = shutil.which('docker')
    if docker_bin is None:
        raise pytest.skip()

    loky_path = os.path.abspath(os.path.dirname(loky.__file__))

    # The following will always run using the Python 3.6 docker image.
    # We mount the loky source as /loky inside the container,
    # so it can be imported when running commands under /
    res = check_output([docker_bin, 'run', '--rm', '--cpus', '0.5',
                        '-v', '%s:/loky' % loky_path,
                        'python:3.6',
                        'python', '-c', cpu_count_cmd.format(args='')])

    assert res.strip().decode('utf-8') == '1'


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

    start_dir = os.path.abspath('.')

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write bad lscpu program
        lscpu_path = tmp_dir + '/lscpu'
        with open(lscpu_path, 'w') as f:
            f.write("#!/bin/sh\n"
                    "exit(1)")
        os.chmod(lscpu_path, 0o777)

        try:
            old_path = os.environ['PATH']
            os.environ['PATH'] = tmp_dir + ":" + old_path

            # clear the cache otherwise the warning is not triggered
            import loky.backend.context
            loky.backend.context.physical_cores_cache = None

            with pytest.warns(UserWarning, match="Could not find the number of"
                                                 " physical cores"):
                cpu_count(only_physical_cores=True)

            # Should not warn the second time
            with pytest.warns(None) as record:
                cpu_count(only_physical_cores=True)
                assert not record

        finally:
            os.environ['PATH'] = old_path


def test_only_physical_cores_with_user_limitation():
    # Check that user limitation for the available number of cores is
    # respected even if only_physical_cores == True
    cpu_count_mp = mp.cpu_count()
    cpu_count_user = _cpu_count_user(cpu_count_mp)

    if cpu_count_user < cpu_count_mp:
        assert cpu_count() == cpu_count_user
        assert cpu_count(only_physical_cores=True) == cpu_count_user
