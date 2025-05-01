r"""The :mod:`loky` module manages a pool of worker that can be re-used across time.
It provides a robust and dynamic implementation os the
:class:`ProcessPoolExecutor` and a function :func:`get_reusable_executor` which
hide the pool management under the hood.
"""

from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    CancelledError,
    Executor,
    TimeoutError,
    as_completed,
    wait,
)

from ._base import Future
from .backend.context import cpu_count
from .backend.spawn import freeze_support
from .backend.reduction import set_loky_pickler
from .reusable_executor import get_reusable_executor
from .cloudpickle_wrapper import wrap_non_picklable_objects
from .process_executor import BrokenProcessPool, ProcessPoolExecutor


__all__ = [
    # Constants
    "ALL_COMPLETED",
    "FIRST_COMPLETED",
    "FIRST_EXCEPTION",
    # Classes
    "Executor",
    "Future",
    "ProcessPoolExecutor",
    # Functions
    "as_completed",
    "cpu_count",
    "freeze_support",
    "get_reusable_executor",
    "set_loky_pickler",
    "wait",
    "wrap_non_picklable_objects",
    # Errors
    "BrokenProcessPool",
    "CancelledError",
    "TimeoutError",
]


__version__ = "3.6.0dev0"
