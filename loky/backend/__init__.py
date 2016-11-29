import sys

import multiprocessing as mp
from multiprocessing import Pipe, Manager
if sys.platform != "win32":
    from .process import PosixLokyProcess as Process

    if sys.version_info < (3, 4):
        from .synchronize import *
        from .queues import Queue, SimpleQueue
    else:
        from multiprocessing import queues
        _ctx = mp.get_context('spawn')

        def Queue(*args, **kwargs):
            return queues.Queue(*args, ctx=_ctx, **kwargs)

        class LokyContext(mp.context.BaseContext):
            _name = 'loky'
            Process = Process
        mp.context._concrete_contexts['loky'] = LokyContext()

else:
    from multiprocessing import Process
    if sys.version_info[:2] < (3, 4):
        from .queues import SimpleQueue, Queue
        from multiprocessing.synchronize import *
    else:
        from multiprocessing import SimpleQueue, Queue

        _ctx = mp.get_context("spawn")

        def LokyContext():
            return _ctx
        mp.context._concrete_contexts['loky'] = LokyContext()

if sys.version_info > (3, 4):
    import os
    from multiprocessing import synchronize

    @staticmethod
    def _make_name():
        name = '/loky-%i-%s' % (os.getpid(), next(synchronize.SemLock._rand))
        return name

    synchronize.SemLock._make_name = _make_name

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

__all__ = ["Process", "Queue", "SimpleQueue", "Lock", "RLock", "Semaphore",
           "BoundedSemaphore", "Condition", "Event", "Pipe", "Manager"]
