import sys

from .process import ExecProcess as Process
from multiprocessing import Pipe

if sys.version_info < (3, 4):
    from .queues import Queue
    from .synchronize import Event
else:
    import multiprocessing as mp
    from multiprocessing import Event
    from multiprocessing import queues

    def Queue(*args, **kwargs):
        return queues.Queue(*args, ctx=mp.get_context('spawn'), **kwargs)
