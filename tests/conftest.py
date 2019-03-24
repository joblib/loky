import sys
import pytest
import logging
import warnings
from multiprocessing.util import log_to_stderr


def pytest_addoption(parser):
    parser.addoption("--loky-verbosity", type=int, default=logging.DEBUG,
                     help="log-level: integer, SUBDEBUG(5) - INFO(20)")
    parser.addoption("--skip-high-memory", action="store_true",
                     help="skip high-memory test to avoid conflict on CI.")
    parser.addoption("--openblas-present", action='store_true',
                     help="Fail test_limit_openblas_threads if BLAS is not "
                     "found")
    parser.addoption("--mkl-present", action='store_true',
                     help="Fail test_limit_mkl_threads if MKL is not "
                     "found")


def log_lvl(request):
    """Choose logging level for multiprocessing"""
    return request.config.getoption("--loky-verbosity")


@pytest.fixture
def openblas_present(request):
    """Fail the test if OpenBLAS is not found"""
    return request.config.getoption("--openblas-present")


@pytest.fixture
def mkl_present(request):
    """Fail the test if MKL is not found"""
    return request.config.getoption("--mkl-present")


def pytest_configure(config):
    """Setup multiprocessing logging for loky testing"""
    if sys.version_info >= (3, 4):
        logging._levelToName[5] = "SUBDEBUG"
    log = log_to_stderr(config.getoption("--loky-verbosity"))
    log.handlers[0].setFormatter(logging.Formatter(
        '[%(levelname)s:%(processName)s:%(threadName)s] %(message)s'))

    warnings.simplefilter('always')

    # When using this option, make sure numpy is accessible
    if config.getoption("--mkl-present"):
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise ImportError("Need 'numpy' with option --mkl-present")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip-high-memory"):
        # --skip-high-memory given in cli: skip high-memory tests
        return
    skip_high_memory = pytest.mark.skip(
        reason="--skip-high-memory option was provided")
    for item in items:
        if "high_memory" in item.keywords:
            item.add_marker(skip_high_memory)
