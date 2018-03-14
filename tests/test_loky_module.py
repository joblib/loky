import os
import sys

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


def test_cpu_count_travis():
    if (os.environ.get("TRAVIS_OS_NAME") is not None
            and sys.version_info >= (3, 4)):
        # default number of available CPU on Travis CI for OSS projects
        assert cpu_count() == 2
    else:
        pytest.skip()
