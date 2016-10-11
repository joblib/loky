import sys

from multiprocessing import Pipe
if sys.platform != "win32":
    from .process import PosixLokyProcess as Process

    if sys.version_info < (3, 4):
        from .synchronize import Event
        from .queues import Queue, SimpleQueue
    else:
        import multiprocessing as mp
        from multiprocessing import Event
        from multiprocessing import queues

        def Queue(*args, **kwargs):
            return queues.Queue(*args, ctx=mp.get_context('spawn'), **kwargs)

else:
    from multiprocessing import Event, Process
    if sys.version_info[:2] > (3, 3):
        from multiprocessing import SimpleQueue, Queue
        from multiprocessing import context, get_context

        context._concrete_contexts['loky'] = get_context('spawn')
    else:
        from .queues import SimpleQueue, Queue
