###############################################################################
# Basic context management with LokyContext and  provides
# compat for UNIX 2.7 and 3.3
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/context.py
#  * Create a context ensuring loky uses only objects that are compatible
#  * Add LokyContext to the list of context of multiprocessing so loky can be
#    used with multiprocessing.set_start_method
#  * Add some compat function for python2.7 and 3.3.
#

import sys
import multiprocessing as mp


if sys.platform == "win32":
    from multiprocessing import Process as LokyProcess
else:
    from .process import PosixLokyProcess as LokyProcess

if sys.version_info > (3, 4):
    from multiprocessing import get_context
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

    BaseContext = object

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

    def get_context(method="loky"):
        if method == "loky":
            return LokyContext()
        elif method == "loky_no_main":
            if sys.platform == "win32":
                return LokyContext()
            return LokyNoMainContext()
        else:
            raise ValueError("Context {} is not implemented.".format(method))


class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""
    _name = 'loky'
    Process = LokyProcess

    def Queue(self, maxsize=0, reducers=None):
        '''Returns a queue object'''
        from .queues import Queue
        return Queue(maxsize, reducers=reducers,
                     ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        '''Returns a queue object'''
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())

    if sys.version_info[:2] < (3, 4):
        """Compat for python2.7/3.3 for necessary methods in Context"""
        def get_context(self):
            return self

        def get_start_method(self):
            return "loky"

        def Pipe(self, duplex=True):
            '''Returns two connection object connected by a pipe'''
            return mp.Pipe(duplex)

        if sys.platform != "win32":
            """Use the compat Manager for python2.7/3.3 on UNIX to avoid
            relying on fork processes
            """
            def Manager(self):
                """Returns a manager object"""
                from .managers import LokyManager
                m = LokyManager()
                m.start()
                return m
        else:
            """Compat for context on Windows and python2.7/3.3. Using regular
            multiprocessing objects as it does not rely on fork.
            """
            from multiprocessing import synchronize
            Semaphore = staticmethod(synchronize.Semaphore)
            BoundedSemaphore = staticmethod(synchronize.BoundedSemaphore)
            Lock = staticmethod(synchronize.Lock)
            RLock = staticmethod(synchronize.RLock)
            Condition = staticmethod(synchronize.Condition)
            Event = staticmethod(synchronize.Event)
            Manager = staticmethod(mp.Manager)

    if sys.platform != "win32":
        """For Unix platform, use our custom implementation of synchronize
        relying on ctypes to interface with pthread semaphores.
        """
        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from . import synchronize
            return synchronize.Semaphore(value=value)

        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value)

        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock
            return Lock()

        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock
            return RLock()

        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition
            return Condition(lock)

        def Event(self):
            """Returns an event object"""
            from .synchronize import Event
            return Event()


class LokyNoMainContext(LokyContext):
    """Extra context with LokyProcess, which do not load the main module

    This context is used for backward compatibility in ``joblib`` and should
    not be used without a good knowledge of the mechanism behind it.
    It alows a script without the ``if __name__ == "__main__":`` to run without
    failure when using ``loky`` backend and ``cloudpickle`` for serialization.
    """
    def Process(self, *args, **kwargs):
        kwargs.pop('init_main_module', False)
        return LokyProcess(*args, init_main_module=False, **kwargs)


if sys.version_info > (3, 4):
    """Register loky context so it works with multiprocessing.get_context"""
    mp.context._concrete_contexts['loky'] = LokyContext()
    if sys.platform == "win32":
        mp.context._concrete_contexts['loky_no_main'] = LokyContext()
    else:
        mp.context._concrete_contexts['loky_no_main'] = LokyNoMainContext()
