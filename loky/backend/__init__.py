import sys

from multiprocessing import Pipe
if sys.platform != "win32":
    from .process import PosixLokyProcess as Process

    if sys.version_info < (3, 4):
        from .synchronize import *
        from .queues import Queue, SimpleQueue
    else:
        import multiprocessing as mp
        from multiprocessing import synchronize
        from multiprocessing import queues
        _ctx = mp.get_context('spawn')

        def Queue(*args, **kwargs):
            return queues.Queue(*args, ctx=_ctx, **kwargs)

        def Semaphore(*args, **kwargs):
            return synchronize.Semaphore(*args, ctx=_ctx, **kwargs)

        def BoundedSemaphore(*args, **kwargs):
            return synchronize.BoundedSemaphore(*args, ctx=_ctx, **kwargs)

        def Lock(*args, **kwargs):
            return synchronize.Lock(*args, ctx=_ctx, **kwargs)

        def RLock(*args, **kwargs):
            return synchronize.RLock(*args, ctx=_ctx, **kwargs)

        def Event(*args, **kwargs):
            return synchronize.Event(*args, ctx=_ctx, **kwargs)

        def Condition(*args, **kwargs):
            return synchronize.Condition(*args, ctx=_ctx, **kwargs)

        class LokyContext(mp.context.BaseContext):
            _name = 'loky'
            Process = Process
        mp.context._concrete_contexts['loky'] = LokyContext()

else:
    from multiprocessing import Process
    from multiprocessing.synchronize import *
    if sys.version_info[:2] > (3, 3):
        from multiprocessing import SimpleQueue, Queue
        from multiprocessing import context, get_context

        _ctx = get_context("spawn")

        def LokyContext():
            return _ctx
        context._concrete_contexts['loky'] = LokyContext()
    else:
        from .queues import SimpleQueue, Queue

__all__ = ["Process", "Queue", "SimpleQueue", "Lock", "RLock", "Semaphore",
           "BoundedSemaphore", "Condition", "Event", "Pipe"]
