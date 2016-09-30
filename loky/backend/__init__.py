import sys

from multiprocessing import Pipe
if sys.platform != "win32":
    from .process import PosixExecProcess as Process

    if sys.version_info < (3, 4):
        from .queues import Queue, SimpleQueue
        from .synchronize import Event
    else:
        import multiprocessing as mp
        from multiprocessing import Event
        from multiprocessing import queues

        def Queue(*args, **kwargs):
            return queues.Queue(*args, ctx=mp.get_context('spawn'), **kwargs)

else:
    from multiprocessing import Queue, Event, Process
    if sys.version_info[:2] > (3, 3):
        from multiprocessing import SimpleQueue
        from multiprocessing import context, get_context

        context._concrete_contexts['loky'] = get_context('spawn')
    else:
        from multiprocessing.queues import SimpleQueue
