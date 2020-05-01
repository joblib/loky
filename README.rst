Reusable Process Pool Executor
==============================
|Build Status| |Build status| |codecov|


Goal
~~~~

The aim of this project is to provide a robust, cross-platform and
cross-version implementation of the :class:`~concurrent.futures.ProcessPoolExecutor` class of
:mod:`concurrent.futures`. It notably features:

-  **Deadlock free implementation**: one of the major concern in standard :class:`multiprocessing.pool.Pool` and in :class:`concurrent.futures.ProcessPoolExecutor` is their ability to handle crashes of worker processes. This library intends to fix those possible deadlocks and send back meaningful errors.

-  **Consistent spawn behavior**: All processes are started using fork/exec on POSIX systems. This ensures safer interactions with third party libraries.

-  **Reusable executor**: strategy to avoid re-spawning a complete executor every time. A singleton executor instance can be reused (and dynamically resized if necessary) across consecutive calls to limit spawning and shutdown overhead. The worker processes can be shutdown automatically after a configurable idling timeout to free system resources.

-  **Transparent** |cloudpickle| **integration**: to call interactively defined functions and lambda expressions in parallel. It is also possible to register a custom pickler implementation to handle inter-process communications.

-  **No need for** :code:`if __name__ == "__main__":` **in scripts**: thanks to the use of |cloudpickle| to call functions defined in the :mod:`__main__` module, it is not required to protect the code calling parallel functions under Windows.


Installation
~~~~~~~~~~~~

The recommended way to install :mod:`loky` is with :mod:`pip`,

.. code:: bash

    pip install loky

:mod:`loky` can also be installed from sources using

.. code:: bash

    git clone https://github.com/tommoral/loky
    cd loky
    python setup.py install

Note that `loky` has an optional dependency on |psutil| to allow early memory leak detections.

Usage
~~~~~

The basic usage of :mod:`loky` relies on the :func:`~loky.get_reusable_executor`, which internally manages a custom :class:`~concurrent.futures.ProcessPoolExecutor` object, which is reused or re-spawned depending on the context.

.. code:: python

    import os
    from time import sleep
    from loky import get_reusable_executor


    def say_hello(k):
        pid = os.getpid()
        print("Hello from {} with arg {}".format(pid, k))
        sleep(.01)
        return pid


    # Create an executor with 4 worker processes, that will
    # automatically shutdown after idling for 2s
    executor = get_reusable_executor(max_workers=4, timeout=2)

    res = executor.submit(say_hello, 1)
    print("Got results:", res.result())

    results = executor.map(say_hello, range(50))
    n_workers = len(set(results))
    print("Number of used processes:", n_workers)
    assert n_workers == 4

For more advance usage, see our documentation_.


Acknowledgement
~~~~~~~~~~~~~~~

This work is supported by the Center for Data Science, funded by the
IDEX Paris-Saclay, ANR-11-IDEX-0003-02


.. |azurepipelines| image:: https://dev.azure.com/joblib/loky/_apis/build/status/joblib.loky?branchName=master
   :target: https://dev.azure.com/joblib/loky/_build?definitionId=2&_a=summary&repositoryFilter=2&branchFilter=38
.. |codecov| image:: https://codecov.io/gh/joblib/loky/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/joblib/loky



.. |cloudpickle| raw:: html

    <a href="https://github.com/cloudpipe/cloudpickle">
        <code>cloudpickle</code>
    </a>

.. |psutil| raw:: html

    <a href="https://github.com/giampaolo/psutil">
        <code>psutil</code>
    </a>

.. _documentation:  http://loky.readthedocs.io/en/stable
