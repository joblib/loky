API Reference
=============


.. automodule:: loky
    :members: get_reusable_executor


Task & results serialization
----------------------------

To share function definition across multiple python processes, it is necessary to rely on a serialization protocol. The standard protocol in python is :mod:`pickle` but its default implementation in the standard library has several limitations. For instance, it cannot serialize functions which are defined interactively or in the :code:`__main__` module.

To avoid this limitation, :mod:`loky` relies on |cloudpickle| when it is present. |cloudpickle| is a fork of the pickle protocol which allows the serialization of a greater number of objects and it can be installed using :code:`pip install cloudpickle`. As this library is slower than the :mod:`pickle` module in the standard library, by default, :mod:`loky` uses it only to serialize objects which are detected to be in the :code:`__main__` module.

There is two way to temper with the serialization in :mod:`loky`:

- Using the arguments :code:`job_reducers` and :code:`result_reducers`, it is possible to register custom reducers for the serialization process.
- Setting the variable :code:`LOKY_PICKLER` to an available and valid serialization module. This module must present a valid :code:`Pickler` object. Setting the environment variable :code:`LOKY_PICKER=cloudpickle` will force :mod:`loky` to serialize everything with |cloudpickle| instead of just serializing the object detected to be in the :code:`__main__` module.


.. |cloudpickle| raw:: html

    <a href="https://github.com/cloudpipe/cloudpickle"><code>cloudpickle</code></a>
