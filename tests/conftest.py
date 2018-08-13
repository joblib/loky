import sys
import pytest
import logging
import warnings
from multiprocessing.util import log_to_stderr


def pytest_addoption(parser):
    parser.addoption("--loky-verbosity", type=int, default=logging.DEBUG,
                     help="log-level: integer, SUBDEBUG(5) - INFO(20)")
    parser.addoption("--skip-memory", action="store_true",
                     help="skip memory test to avoid conflict on CI.")


def log_lvl(request):
    """Choose logging level for multiprocessing"""
    return request.config.getoption("--loky-verbosity")


def pytest_configure(config):
    """Setup multiprocessing logging for loky testing"""
    if sys.version_info >= (3, 4):
        logging._levelToName[5] = "SUBDEBUG"
    log = log_to_stderr(config.getoption("--loky-verbosity"))
    log.handlers[0].setFormatter(logging.Formatter(
        '[%(levelname)s:%(processName)s:%(threadName)s] %(message)s'))

    warnings.simplefilter('always')


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip-memory"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_memory = pytest.mark.skip(reason="--skip-memory option was provided")
    for item in items:
        if "memory" in item.keywords:
            item.add_marker(skip_memory)
