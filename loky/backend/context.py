import sys
import multiprocessing as mp


if sys.platform == "win32":
    from multiprocessing import Process
else:
    from .process import PosixLokyProcess as Process

if sys.version_info > (3, 4):
    from multiprocessing.context import assert_spawning, set_spawning_popen
    from multiprocessing.context import get_spawning_popen, BaseContext

else:
    if sys.platform != 'win32':
        import threading
        # Mecanism to check that the current thread is spawning a child process
        _tls = threading.local()
        popen_attr = 'spawning_popen'
    else:
        from multiprocessing.forking import Popen
        _tls = Popen._tls
        popen_attr = 'process_handle'

    def get_spawning_popen():
        return getattr(_tls, popen_attr, None)

    def set_spawning_popen(popen):
        setattr(_tls, popen_attr, popen)

    def assert_spawning(obj):
        if get_spawning_popen() is None:
            raise RuntimeError(
                '%s objects should only be shared between processes'
                ' through inheritance' % type(obj).__name__
            )

    BaseContext = object


class LokyContext(BaseContext):
    _name = 'loky'
    Process = Process

    def Queue(self, maxsize=0, reducers=None):
        '''Returns a queue object'''
        from .queues import Queue
        return Queue(maxsize, reducers=reducers,
                     ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        '''Returns a queue object'''
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())
    if sys.platform != "win32":
        def Semaphore(self, value=1):
            from . import synchronize
            return synchronize.Semaphore(value=value)

        def BoundedSemaphore(self, value):
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value)

        def Lock(self):
            from .synchronize import Lock
            return Lock()

        def RLock(self):
            from .synchronize import RLock
            return RLock()

        def Condition(self, lock=None):
            from .synchronize import Condition
            return Condition(lock)

        def Event(self):
            from .synchronize import Event
            return Event()
    else:
        if sys.version_info[:2] < (3, 4):
            from multiprocessing import synchronize
            Semaphore = synchronize.Semaphore
            BoundedSemaphore = synchronize.BoundedSemaphore
            Lock = synchronize.Lock
            RLock = synchronize.RLock
            Condition = synchronize.Condition
            Event = synchronize.Event

    if sys.version_info[:2] < (3, 4):
        def get_context(self):
            return self

        def Manager(self):
            if sys.platform == "win32":
                return mp.Manager()
            from .managers import LokyManager
            m = LokyManager()
            m.start()
            return m

        def Pipe(self, duplex=True):
            '''Returns two connection object connected by a pipe'''
            return mp.Pipe(duplex)


if sys.version_info > (3, 4):
    mp.context._concrete_contexts['loky'] = LokyContext()
    from multiprocessing import get_context

else:
    def get_context():
        return LokyContext()
