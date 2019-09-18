import os
import sys
import shutil
from subprocess import check_output

import pytest

import loky
from loky import cpu_count


def test_version():
    assert hasattr(loky, '__version__'), (
        "There are no __version__ argument on the loky module")


def test_cpu_count():
    cpus = cpu_count()
    assert type(cpus) is int
    assert cpus >= 1


cpu_count_cmd = ("from loky.backend.context import cpu_count;"
                 "print(cpu_count())")


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
                        python_bin, '-c', cpu_count_cmd])

    assert res.strip().decode('utf-8') == '1'


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
                        'python', '-c', cpu_count_cmd])

    assert res.strip().decode('utf-8') == '1'
