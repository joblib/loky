import os
import sys
from multiprocessing import synchronize

from .context import get_context


def _make_name():
    name = f'/loky-{os.getpid()}-{next(synchronize.SemLock._rand)}'
    return name

# monkey patch the name creation for multiprocessing
synchronize.SemLock._make_name = staticmethod(_make_name)

__all__ = ["get_context"]
