import os
import sys
import shutil
from subprocess import check_output

import pytest

import loky
from loky.backend.context import cpu_count


def test_version():
    assert hasattr(loky, '__version__'), (
        "There are no __version__ argument on the loky module")


def test_cpu_count():
    cpus = cpu_count()
    assert type(cpus) is int
    assert cpus >= 1


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

    cmd = "from loky.backend.context import cpu_count; print(cpu_count())"

    res = check_output([taskset_bin, '-c', '1', python_bin, '-c', cmd])

    assert res.strip().decode('utf-8') == '1'


def test_cpu_count_travis():
    if (os.environ.get("TRAVIS_OS_NAME") is not None
            and sys.version_info >= (3, 4)):
        # default number of available CPU on Travis CI for OSS projects
        assert cpu_count() == 2
    else:
        pytest.skip()
