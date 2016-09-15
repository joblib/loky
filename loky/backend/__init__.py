import sys

from .process import ExecProcess as Process
from multiprocessing import Pipe, Event

if sys.version_info < (3, 4):
    from .queues import Queue
else:
    import multiprocessing as mp
    from multiprocessing import queues

    def Queue(*args, **kwargs):
        return queues.Queue(*args, ctx=mp.get_context('spawn'), **kwargs)
