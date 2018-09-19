import inspect
from functools import partial

from .backend import LOKY_PICKLER

try:
    from cloudpickle import dumps, loads
    cloudpickle = True
except ImportError:
    cloudpickle = False


WRAP_CACHE = dict()


class CloudpickledObjectWrapper(object):
    def __init__(self, obj, keep_wrapper=False):
        self._obj = obj
        self._keep_wrapper = keep_wrapper

    def __reduce__(self):
        _pickled_object = dumps(self._obj)
        if not self._keep_wrapper:
            return loads, (_pickled_object,)

        return _reconstruct_wrapper, (_pickled_object, self._keep_wrapper)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seemlessly as the
        # previous object.
        if attr not in ['__reduce__', '_obj', '_keep_wrapper']:
            return getattr(self._obj, attr)
        return getattr(self, attr)


def _reconstruct_wrapper(_pickled_object, keep_wrapper):
    return CloudpickledObjectWrapper(loads(_pickled_object), keep_wrapper)


def _wrap_objects_when_needed(obj):
    if LOKY_PICKLER or not cloudpickle:
        return obj

    need_wrap = "__main__" in getattr(obj, "__module__", "")
    if isinstance(obj, partial):
        return partial(
            _wrap_objects_when_needed(obj.func),
            *[_wrap_objects_when_needed(a) for a in obj.args],
            **{k: _wrap_objects_when_needed(v)
                for k, v in obj.keywords.items()}
        )
    if callable(obj):
        # Need wrap if the object is a function defined in a local scope of
        # another function.
        func_code = getattr(obj, "__code__", "")
        need_wrap |= getattr(func_code, "co_flags", 0) & inspect.CO_NESTED

        # Need wrap if the obj is a lambda expression
        func_name = getattr(obj, "__name__", "")
        need_wrap |= "<lambda>" in func_name

    if not need_wrap:
        return obj

    wrapped_obj = WRAP_CACHE.get(obj)
    if wrapped_obj is None:
        wrapped_obj = CloudpickledObjectWrapper(obj, keep_wrapper=False)
        WRAP_CACHE[obj] = wrapped_obj
    return wrapped_obj


def wrap_non_picklable_objects(obj):
    """Wrapper for non-picklable object to use cloudpickle to serialize them.

    Note that this wrapper tends to slow down the serialization process has it
    is done with cloudpickle which is typically slower compared to pickle. The
    proper way to solve serialization issue is to avoid defining functions and
    objects in the main scripts and to implement __reduce__ functions for
    complex classes.
    """
    if not cloudpickle:
        raise ImportError("could not import cloudpickle. Please install "
                          "cloudpickle to allow extended serialization. "
                          "(`pip install cloudpickle`).")

    # If obj is an  class, create a CloudpickledClassWrappers which instantiate
    # the object internally and wrap it directly in a CloudpickledObjectWrapper
    if inspect.isclass(obj):
        class CloudpickledClassWrapper(CloudpickledObjectWrapper):
            def __init__(self, *args, **kwargs):
                self._obj = obj(*args, **kwargs)
                self._keep_wrapper = True

        return CloudpickledClassWrapper

    # If obj is an instance of a class, just wrap it in a regular
    # CloudpickledObjectWrapper
    return CloudpickledObjectWrapper(obj, keep_wrapper=True)
