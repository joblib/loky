# -*- coding: utf-8 -*-
"""
Serialization of un-picklable objects
=====================================

This example highlights the options for tempering with loky serialization
process.

"""

# Code source: Thomas Moreau
# License: BSD 3 clause

import sys
import time
import traceback
from loky import set_loky_pickler
from loky import get_reusable_executor
from loky import wrap_non_picklable_objects

###############################################################################
# First, define functions which cannot be pickled with the standard ``pickle``
# protocol. They cannot be serialized with ``pickle`` because they are defined
# in the ``__main__`` module. They can however be serialized with
# ``cloudpickle``.
#


def call_function(list_or_func, x, *args):
    while isinstance(list_or_func, list):
        list_or_func = list_or_func[0]
    return list_or_func(x)


def func_async(i):
    return 2 * i


###############################################################################
# With the default behavior, ``loky`` is able to detect that this function is
# in the ``__main__`` module and internally use a wrapper with ``cloudpickle``
# to serialize it.
#

executor = get_reusable_executor(max_workers=1)
print(executor.submit(call_function, func_async, 21).result())

###############################################################################
# However, the mechanism to detect that the wrapper is needed fails when this
# function is nested in objects that are picklable. For instance, if this
# function is given in a list of list, loky won't be able to wrap it and the
# serialization of the task will fail.
#

try:
    executor = get_reusable_executor(max_workers=1)
    executor.submit(id, [[func_async]]).result()
except Exception:
    traceback.print_exc(file=sys.stdout)


###############################################################################
# Other failures include ``dict`` with non-serializable objects or a
# serializable object containing a field with non-serializable objects.
#
# To avoid this, it is possible to fully rely on ``cloudpickle`` to serialize
# all communications between the main process and the workers. This can be done
# with an environment variable ``LOKY_PICKLER=cloudpickle`` set before the
# script is launched, or with the switch ``set_loky_pickler`` provided in the
# ``loky`` API.
#

set_loky_pickler('cloudpickle')
executor = get_reusable_executor(max_workers=1)
print(executor.submit(call_function, [[func_async]], 21).result())


###############################################################################
# For most use-cases, using ``cloudpickle``` is sufficient. However, this
# solution can be slow to serialize large python objects, such as dict or list.
#

# We have to pass an extra argument with a large list (or another large python
# object).
large_list = list(range(1000000))

# We are still using ``cloudpickle`` to serialize the task as we did not reset
# the loky_pickler.
t_start = time.time()
executor = get_reusable_executor(max_workers=1)
executor.submit(call_function, [[func_async]], 21, large_list).result()
print("With cloudpickle serialization: {:.3f}s".format(time.time() - t_start))

# Now reset the `loky_pickler` to the default behavior. Note that here, we do
# not pass the desired function ``func_async`` as it is not picklable but it is
# replaced by ``id`` for demonstration purposes.
set_loky_pickler()
t_start = time.time()
executor = get_reusable_executor(max_workers=1)
executor.submit(call_function, [[id]], 21, large_list).result()
print("With default serialization: {:.3f}s".format(time.time() - t_start))


###############################################################################
# To avoid this performance drop, it is possible to wrap the non-picklable
# function using :func:`wrap_non_picklable_objects` to indicate to the
# serialization process that this object should be serialized using
# ``cloudpickle``. This changes the serialization behavior only for this
# function and keeps the default behavior for all other objects. The drawback
# of this solution is that it modifies the object. This should not cause many
# issues with functions but can have side effects with object instances.
#

@wrap_non_picklable_objects
def func_async(i):
    return 2 * i


t_start = time.time()
executor = get_reusable_executor(max_workers=1)
executor.submit(call_function, [[func_async]], 21, large_list).result()
print("With default and wrapper: {:.3f}s".format(time.time() - t_start))


###############################################################################
# The same wrapper can also be used for non-picklable classes. Note that the
# side effects of :func:`wrap_non_picklable_objects` on objects can break magic
# methods such as ``__add__`` and can mess up the ``isinstance`` and
# ``issubclass`` functions. Some improvements will be considered if use-cases
# are reported.
#
