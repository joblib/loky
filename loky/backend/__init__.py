import sys

from .process import ExecProcess as Process

if sys.version_info < (3, 4):
    from .queues import Queue, SimpleQueue
    from .synchronize import Event
    # from .connection import Pipe
    from multiprocessing import Pipe
else:
    import multiprocessing as mp
    from multiprocessing import Event, Pipe
    from multiprocessing import queues

    def Queue(*args, **kwargs):
        return queues.Queue(*args, ctx=mp.get_context('spawn'), **kwargs)
