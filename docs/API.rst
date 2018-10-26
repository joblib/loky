API Reference
=============


.. automodule:: loky
    :members: get_reusable_executor


Task & results serialization
----------------------------

To share function definition across multiple python processes, it is necessary to rely on a serialization protocol. The standard protocol in python is :mod:`pickle` but its default implementation in the standard library has several limitations. For instance, it cannot serialize functions which are defined interactively or in the :code:`__main__` module.

To avoid this limitation, :mod:`loky` relies on |cloudpickle| when it is present. |cloudpickle| is a fork of the pickle protocol which allows the serialization of a greater number of objects and it can be installed using :code:`pip install cloudpickle`. As this library is slower than the :mod:`pickle` module in the standard library, by default, :mod:`loky` uses it only to serialize objects which are detected to be in the :code:`__main__` module.

There are three ways to temper with the serialization in :mod:`loky`:

- Using the arguments :code:`job_reducers` and :code:`result_reducers`, it is possible to register custom reducers for the serialization process.
- Setting the variable :code:`LOKY_PICKLER` to an available and valid serialization module. This module must present a valid :code:`Pickler` object. Setting the environment variable :code:`LOKY_PICKER=cloudpickle` will force :mod:`loky` to serialize everything with |cloudpickle| instead of just serializing the object detected to be in the :code:`__main__` module.
- Finally, it is possible to wrap an unpicklable object using the :code:`loky.wrap_non_picklable_objects` decorator. In this case, all other objects are serialized as in the default behavior and the wrapped object is pickled through |cloudpickle|.

The benefits and drawbacks of each method are highlighted in this example_.


.. autofunction:: loky.wrap_non_picklable_objects


Processes start methods in :mod:`loky`
--------------------------------------

The API in :mod:`loky` provides a :func:`set_start_method` function to set the default  :code:`start_method`, which controls the way :class:`Process` are started. The available methods are {:code:`'loky'`, :code:`'loky_int_main'`, :code:`'spawn'`}. On unix, the start methods {:code:`'fork'`, :code:`'forkserver'`} are also available.
Note that :mod:`loky` isnot compatible with :func:`multiprocessing.set_start_method` function. The default start method needs to be set with the provided function to ensure a proper behavior.


Protection against memory leaks
-------------------------------
The memory size of long running worker processes can increase indefinitely if a
memory leak is created. This can result in processes being shut down by the OS if
those leaks are not resolved. To
prevent it, loky provides leak detection, memory cleanups, and workers 
shutdown.

If :mod:`psutil` is installed, each worker periodically [#periodically_fn]_ checks its 
memory usage after it completes its task. If the usage is found to be
unusual [#psutil_unusual_fn]_, an additional :code:`gc.collect()` event is triggered to remove 
objects with potential cyclic references. 
If even after that, the memory usage of a process worker remains too high, 
it will shut down safely, and a fresh process will be automatically spawned by
the executor. 

If :mod:`psutil` is not installed, there is no easy way to monitor worker
processes memory usage. :code:`gc.collect()` events will still be called
periodially [#periodically_fn]_ inside each workers, but there is no guarantee that a leak is
not happening.

.. rubric:: Footnotes

.. [#periodically_fn] every 1 second. This constant is define in :code:`loky.process_executor._MEMORY_LEAK_CHECK_DELAY`
.. [#psutil_unusual_fn] an increase of 100MB compared to a reference, which is defined as the residual memory usage of the worker after it completed its first task



.. |cloudpickle| raw:: html

    <a href="https://github.com/cloudpipe/cloudpickle"><code>cloudpickle</code></a>

.. _example :  auto_examples/cloudpickle_wrapper.html