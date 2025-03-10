.. raw:: html

    <img alt="Logo loky" src="_static/loky_logo.svg" width="128px"
     style="float: right;">

Reusable Process Pool Executor
==============================
|azurepipelines| |codecov|


Goal
~~~~

The aim of this project is to provide a robust, cross-platform and
cross-version implementation of the :class:`~concurrent.futures.ProcessPoolExecutor` class of
:mod:`concurrent.futures`. It notably features:

-  **Deadlock free implementation**: one of the major concern in standard :class:`multiprocessing.pool.Pool` and in :class:`concurrent.futures.ProcessPoolExecutor` is their ability to handle crashes of worker processes. This library intends to fix those possible deadlocks and send back meaningful errors. Note that several fixes in ``loky`` have been ported to :class:`concurrent.futures.ProcessPoolExecutor` since python3.7+, which now be as ``loky``'s implementation.

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

    git clone https://github.com/joblib/loky
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
        print(f"Hello from {pid} with arg {k}")
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


Workflow to contribute
~~~~~~~~~~~~~~~~~~~~~~

To contribute to :mod:`loky`, first create an account on github_. Once this is done, fork the `loky repository`_ to have your own repository, clone it using 'git clone' on the computers where you want to work. Make your changes in your clone, push them to your github account, test them on several computers, and when you are happy with them, send a pull request to the main repository.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

To run the test suite, you need the |pytest| (version >= 3) and |psutil|
modules. From the root of the project, run the test suite using:

.. code:: bash

    pip install -e .
    pytest .


Why was the project named `loky`?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While developping :mod:`loky`, we had some bad experiences trying to debug  deadlocks when using :class:`multiprocessing.pool.Pool` and :class:`concurrent.futures.ProcessPoolExecutor`, especially when calling functions with non-picklable arguments or returned values at the beginning of the project. When we had to chose a name, we had dealt with so many deadlocks that we wanted some kind of invocation to repel them! Hence :mod:`loky`: a mix of a god, locks and the `y` that make it somehow cooler and nicer :) (and also less likely to result in name conflict in google results ^^).

Fixes to avoid those deadlocks in :mod:`concurrent.futures` were also contributed upstream in Python 3.7+, as a less mystical way to repel the deadlocks :D

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

.. |pytest| raw:: html

    <a href="https://pytest.org">
        <code>pytest</code>
    </a>

.. _github: http://github.com/

.. _`loky repository`: http://github.com/joblib/loky

.. _documentation:  http://loky.readthedocs.io/en/stable
