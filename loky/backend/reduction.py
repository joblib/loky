import io
import os
import sys
import functools
try:
    # Python 2 compat
    from cPickle import loads
except ImportError:
    from pickle import loads
    import copyreg

from pickle import Pickler, HIGHEST_PROTOCOL


def _mk_inheritable(fd):
    if sys.version_info[:2] > (3, 3):
        if sys.platform == 'win32':
            # Change to Windwos file handle
            import msvcrt
            fdh = msvcrt.get_osfhandle(fd)
            os.set_handle_inheritable(fdh, True)
            return fdh
        else:
            os.set_inheritable(fd, True)
            return fd
    elif sys.platform == 'win32':
        # TODO: find a hack??
        # Not yet working
        import msvcrt
        import _subprocess

        curproc = _subprocess.GetCurrentProcess()
        fdh = msvcrt.get_osfhandle(fd)
        fdh = _subprocess.DuplicateHandle(
            curproc, fdh, curproc, 0,
            True,  # set inheritable FLAG
            _subprocess.DUPLICATE_SAME_ACCESS)
        return fdh
    else:
        return fd


###############################################################################
# Enable custom pickling in Loky.
# To allow instance customization of the pickling process, we use 2 classes.
# _LokyPickler gives module level customization and CustomizablePickler permits
# to use instance base custom reducers.  Only CustomizablePickler should be use

class _LokyPickler(Pickler):
    """Pickler that uses custom reducers.

    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.

    """

    # We override the pure Python pickler as its the only way to be able to
    # customize the dispatch table without side effects in Python 2.6
    # to 3.2. For Python 3.3+ leverage the new dispatch_table
    # feature from http://bugs.python.org/issue14166 that makes it possible
    # to use the C implementation of the Pickler which is faster.

    if hasattr(Pickler, 'dispatch'):
        # Make the dispatch registry an instance level attribute instead of
        # a reference to the class dictionary under Python 2
        dispatch = Pickler.dispatch.copy()
    else:
        # Under Python 3 initialize the dispatch table with a copy of the
        # default registry
        dispatch_table = copyreg.dispatch_table.copy()

    def __init__(self, writer, protocol=HIGHEST_PROTOCOL):
        Pickler.__init__(self, writer, protocol=protocol)

    @classmethod
    def register(cls, type, reduce_func):
        """Attach a reducer function to a given type in the dispatch table."""
        if hasattr(Pickler, 'dispatch'):
            # Python 2 pickler dispatching is not explicitly customizable.
            # Let us use a closure to workaround this limitation.
            def dispatcher(cls, obj):
                reduced = reduce_func(obj)
                cls.save_reduce(obj=obj, *reduced)
            cls.dispatch[type] = dispatcher
        else:
            cls.dispatch_table[type] = reduce_func


register = _LokyPickler.register


class CustomizableLokyPickler(Pickler):
    def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
        Pickler.__init__(self, writer, protocol=protocol)
        if reducers is None:
            reducers = {}
        if hasattr(Pickler, 'dispatch'):
            # Make the dispatch registry an instance level attribute instead of
            # a reference to the class dictionary under Python 2
            self.dispatch = _LokyPickler.dispatch.copy()
        else:
            # Under Python 3 initialize the dispatch table with a copy of the
            # default registry
            self.dispatch_table = _LokyPickler.dispatch_table.copy()
        for type, reduce_func in reducers.items():
            self.register(type, reduce_func)

    def register(self, type, reduce_func):
        """Attach a reducer function to a given type in the dispatch table."""
        if hasattr(Pickler, 'dispatch'):
            # Python 2 pickler dispatching is not explicitly customizable.
            # Let us use a closure to workaround this limitation.
            def dispatcher(self, obj):
                reduced = reduce_func(obj)
                self.save_reduce(obj=obj, *reduced)
            self.dispatch[type] = dispatcher
        else:
            self.dispatch_table[type] = reduce_func

    @classmethod
    def loads(self, buf):
        if sys.version_info < (3, 3) and isinstance(buf, io.BytesIO):
            buf = buf.getvalue()
        return loads(buf)

    @classmethod
    def dumps(cls, obj, reducers=None, protocol=None):
        buf = io.BytesIO()
        cls(buf, reducers=reducers, protocol=protocol).dump(obj)
        if sys.version_info < (3, 3):
            return buf.getvalue()
        return buf.getbuffer()


def dump(obj, file, reducers=None, protocol=None):
    '''Replacement for pickle.dump() using LokyPickler.'''
    CustomizableLokyPickler(file, reducers=reducers,
                            protocol=protocol).dump(obj)


# make methods picklable
def _reduce_method(m):
    if m.__self__ is None:
        return getattr, (m.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)


class _C:
    def f(self):
        pass

    @classmethod
    def h(cls):
        pass


register(type(_C().f), _reduce_method)
register(type(_C.h), _reduce_method)


def _reduce_method_descriptor(m):
    return getattr, (m.__objclass__, m.__name__)


register(type(list.append), _reduce_method_descriptor)
register(type(int.__add__), _reduce_method_descriptor)


# Make partial func pickable
def _reduce_partial(p):
    return _rebuild_partial, (p.func, p.args, p.keywords or {})


def _rebuild_partial(func, args, keywords):
    return functools.partial(func, *args, **keywords)


register(functools.partial, _reduce_partial)

if sys.platform == "win32":
    from . import _win_reduction  # noqa: F401
else:
    from . import _posix_reduction  # noqa: F401
