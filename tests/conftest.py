import sys
import logging
import warnings
from multiprocessing.util import log_to_stderr


def pytest_addoption(parser):
    parser.addoption("--loky-verbosity", type=int, default=logging.DEBUG,
                     help="log-level: integer, SUBDEBUG(5) - INFO(20)")


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
