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

    config.addinivalue_line("markers", "timeout")
    config.addinivalue_line("markers", "broken_pool")
    config.addinivalue_line("markers", "high_memory")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip-high-memory"):
        # --skip-high-memory given in cli: skip high-memory tests
        return
    skip_high_memory = pytest.mark.skip(
        reason="--skip-high-memory option was provided")
    for item in items:
        if "high_memory" in item.keywords:
            item.add_marker(skip_high_memory)
