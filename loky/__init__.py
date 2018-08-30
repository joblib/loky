r"""The :mod:`loky` module manages a pool of worker that can be re-used across time.
It provides a robust and dynamic implementation os the
:class:`ProcessPoolExecutor` and a function :func:`get_reusable_executor` which
hide the pool management under the hood.
"""
from .reusable_executor import get_reusable_executor  # noqa: F401
from .process_executor import ProcessPoolExecutor  # noqa: F401
from .process_executor import BrokenProcessPool  # noqa: F401

from .backend.context import cpu_count  # noqa: F401

try:
    from concurrent.futures._base import (FIRST_COMPLETED,
                                          FIRST_EXCEPTION,
                                          ALL_COMPLETED,
                                          CancelledError,
                                          TimeoutError,
                                          InvalidStateError,
                                          BrokenExecutor,
                                          Future,
                                          Executor,
                                          wait,
                                          as_completed)
except ImportError:
    from ._base import (FIRST_COMPLETED,
                        FIRST_EXCEPTION,
                        ALL_COMPLETED,
                        CancelledError,
                        TimeoutError,
                        InvalidStateError,
                        BrokenExecutor,
                        Future,
                        Executor,
                        wait,
                        as_completed)

__version__ = '2.3.0.dev0'
