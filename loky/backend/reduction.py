###############################################################################
# Customizable Pickler with some basic reducers
#
# author: Thomas Moreau
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Replace the ForkingPickler with a similar _LokyPickler,
#  * Add CustomizableLokyPickler to allow customizing pickling process
#    on the fly.
#
import io
import sys
import functools
import warnings
from multiprocessing import util
try:
    # Python 2 compat
    from cPickle import loads as pickle_loads
except ImportError:
    from pickle import loads as pickle_loads
    import copyreg

from pickle import HIGHEST_PROTOCOL
from . import LOKY_PICKLER

if sys.platform == "win32":
    if sys.version_info[:2] > (3, 3):
        from multiprocessing.reduction import duplicate
    else:
        from multiprocessing.forking import duplicate


###############################################################################
# Enable custom pickling in Loky.
# To allow instance customization of the pickling process, we use 2 classes.
# _ReducerRegistry gives module level customization and CustomizablePickler
# permits to use instance base custom reducers. Only CustomizablePickler
# should be used.

class _ReducerRegistry(object):
    """Registry for custom reducers.

    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.

    """

    # We override the pure Python pickler as its the only way to be able to
    # customize the dispatch table without side effects in Python 2.6
    # to 3.2. For Python 3.3+ leverage the new dispatch_table
    # feature from http://bugs.python.org/issue14166 that makes it possible
    # to use the C implementation of the Pickler which is faster.

    dispatch_table = {}

    @classmethod
    def register(cls, type, reduce_func):
        """Attach a reducer function to a given type in the dispatch table."""
        if sys.version_info < (3,):
            # Python 2 pickler dispatching is not explicitly customizable.
            # Let us use a closure to workaround this limitation.
            def dispatcher(cls, obj):
                reduced = reduce_func(obj)
                cls.save_reduce(obj=obj, *reduced)
            cls.dispatch[type] = dispatcher
        else:
            cls.dispatch_table[type] = reduce_func


###############################################################################
# Registers extra pickling routines to improve picklization  for loky

register = _ReducerRegistry.register


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


if not hasattr(sys, "pypy_version_info"):
    # PyPy uses functions instead of method_descriptors and wrapper_descriptors
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

if sys.platform != "win32":
    from ._posix_reduction import _mk_inheritable  # noqa: F401
else:
    from . import _win_reduction  # noqa: F401

# global variable to change the pickler behavior
_LokyPickler = None
_use_cloudpickle_wrapper = True


def set_loky_pickler(pickler=None):
    global _LokyPickler, _use_cloudpickle_wrapper

    if pickler is None:
        pickler = LOKY_PICKLER

    pickler_cls = None

    if pickler in ["pickle", "", None]:
        # Only use the cloudpickle_wrapper when pickler is None or ''
        from pickle import Pickler as pickler_cls
        _use_cloudpickle_wrapper = pickler != "pickle"
    else:
        _use_cloudpickle_wrapper = False
        try:
            if pickler == 'cloudpickle':
                from cloudpickle import CloudPickler as pickler_cls
            else:
                from importlib import import_module
                module_pickle = import_module(pickler)
                if not hasattr(module_pickle, 'Pickler'):
                    raise ValueError("Failed to find Pickler object in module "
                                     "'{}', requested for pickling."
                                     .format(pickler))
                pickler_cls = module_pickle.Pickler
        except ImportError:
            raise ValueError("Failed to import '{}' requested for pickler. "
                             "Make sure it is available on your system."
                             .format(pickler))

    util.debug("Using default backend {} for pickling."
               .format(pickler if pickler else "pickle"))

    class CustomizablePickler(pickler_cls):

        if sys.version_info < (3,):
            # Make the dispatch registry an instance level attribute instead of
            # a reference to the class dictionary under Python 2
            _dispatch = pickler_cls.dispatch.copy()
            _dispatch.update(_ReducerRegistry.dispatch_table)
        else:
            # Under Python 3 initialize the dispatch table with a copy of the
            # default registry
            _dispatch_table = copyreg.dispatch_table.copy()
            _dispatch_table.update(_ReducerRegistry.dispatch_table)

        def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
            pickler_cls.__init__(self, writer, protocol=protocol)
            if reducers is None:
                reducers = {}
            if sys.version_info < (3,):
                self.dispatch = self._dispatch.copy()
            else:
                self.dispatch_table = self._dispatch_table.copy()
            for type, reduce_func in reducers.items():
                self.register(type, reduce_func)

        def register(self, type, reduce_func):
            """Attach a reducer function to a given type in the dispatch table."""
            util.debug((type, "register", reduce_func))
            if sys.version_info < (3,):
                # Python 2 pickler dispatching is not explicitly customizable.
                # Let us use a closure to workaround this limitation.
                    def dispatcher(self, obj):
                        reduced = reduce_func(obj)
                        self.save_reduce(obj=obj, *reduced)
                    self.dispatch[type] = dispatcher
            else:
                self.dispatch_table[type] = reduce_func

    _LokyPickler = CustomizablePickler


def use_cloudpickle_wrapper():
    return _use_cloudpickle_wrapper


# Set it to its default value
set_loky_pickler()


def loads(buf):
    # Compat for python2.7 version
    if sys.version_info < (3, 3) and isinstance(buf, io.BytesIO):
        buf = buf.getvalue()
    return pickle_loads(buf)


def dump(obj, file, reducers=None, protocol=None):
    '''Replacement for pickle.dump() using _LokyPickler.'''
    global _LokyPickler
    pickler = _LokyPickler(file, reducers=reducers, protocol=protocol)

    from pickle import Pickler
    assert issubclass(_LokyPickler, Pickler)
    if reducers is None:
        reducers = {}
    for type, reduce_func in reducers.items():
        pickler.dispatch_table[type] = reduce_func
    pickler.dump(obj)


def dumps(obj, reducers=None, protocol=None):
    global _LokyPickler

    buf = io.BytesIO()
    dump(obj, buf, reducers=reducers, protocol=protocol)
    if sys.version_info < (3, 3):
        return buf.getvalue()
    return buf.getbuffer()


__all__ = ["dump", "dumps", "loads", "register",
           "set_loky_pickler", "use_cloudpickle_wrapper"]

if sys.platform == "win32":
    __all__ += ["duplicate"]
