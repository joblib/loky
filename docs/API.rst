API Reference
=============

.. automodule:: loky
    :members: get_reusable_executor, cpu_count

.. autofunction:: loky.backend.utils.limit_threads_clib


Task & results serialization
----------------------------

To share function definition across multiple python processes, it is necessary to rely on a serialization protocol. The standard protocol in python is :mod:`pickle` but its default implementation in the standard library has several limitations. For instance, it cannot serialize functions which are defined interactively or in the :code:`__main__` module.

To avoid this limitation, :mod:`loky` relies on |cloudpickle| when it is present. |cloudpickle| is a fork of the pickle protocol which allows the serialization of a greater number of objects and it can be installed using :code:`pip install cloudpickle`. As this library is slower than the :mod:`pickle` module in the standard library, by default, :mod:`loky` uses it only to serialize objects which are detected to be in the :code:`__main__` module.

There is two way to temper with the serialization in :mod:`loky`:

- Using the arguments :code:`job_reducers` and :code:`result_reducers`, it is possible to register custom reducers for the serialization process.
- Setting the variable :code:`LOKY_PICKLER` to an available and valid serialization module. This module must present a valid :code:`Pickler` object. Setting the environment variable :code:`LOKY_PICKER=cloudpickle` will force :mod:`loky` to serialize everything with |cloudpickle| instead of just serializing the object detected to be in the :code:`__main__` module.


Processes start methods in :mod`loky`
-------------------------------------

The API in :mod:`loky` provides a :func:`set_start_method` function to set the default  :code:`start_method`, which controls the way :class:`Process` are started. The available methods are {:code:`'loky'`, :code:`'loky_int_main'`, :code:`'spawn'`}. On unix, the start methods {:code:`'fork'`, :code:`'forkserver'`} are also available.
Note that :mod:`loky` isnot compatible with :func:`multiprocessing.set_start_method` function. The default start method needs to be set with the provided function to ensure a proper behavior.


.. |cloudpickle| raw:: html

    <a href="https://github.com/cloudpipe/cloudpickle"><code>cloudpickle</code></a>
