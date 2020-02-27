###############################################################################
# Compat file to import the correct modules for each platform and python
# version.
#
# author: Thomas Moreau and Olivier grisel
#
import queue

from multiprocessing.process import BaseProcess
from multiprocessing.connection import wait


PY3 = True


def set_cause(exc, cause):
    exc.__cause__ = cause

    if not PY3:
        # Preformat message here.
        if exc.__cause__ is not None:
            exc.args = ("{}\n\nThis was caused directly by {}".format(
                exc.args if len(exc.args) != 1 else exc.args[0],
                str(exc.__cause__)),)

    return exc


__all__ = ["queue", "BaseProcess", "set_cause", "wait"]
