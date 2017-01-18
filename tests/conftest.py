import sys
import pytest
import logging
from multiprocessing.util import log_to_stderr


def pytest_addoption(parser):
    parser.addoption("--verbosity", type=int, default=logging.DEBUG,
                     help="log-level: integer, SUBDEBUG(5) - INFO(20)")


@pytest.fixture(scope="session")
def log_lvl(request):
    """Choose logging level for multiprocessing"""
    return request.config.getoption("--verbosity")


@pytest.yield_fixture(scope="session", autouse=True)
def logging_setup(log_lvl):
    """Setup multiprocessing logging for loky testing"""
    if sys.version_info >= (3, 4):
        logging._levelToName[5] = "SUBDEBUG"
    log = log_to_stderr(log_lvl)
    log.handlers[0].setFormatter(logging.Formatter(
        '[%(levelname)s:%(processName)s:%(threadName)s] %(message)s'))
    yield
    del log.handlers[:]
