
<a href="https://loky.readthedocs.io">
<img src="docs/_static/loky_logo.svg"
alt="Loky logo" width=96/></a>


# Reusable Process Pool Executor
[![Build Status](https://github.com/joblib/loky/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/joblib/loky/actions/workflows/test.yml?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/loky/badge/?version=latest)](https://loky.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/joblib/loky/branch/master/graph/badge.svg)](https://codecov.io/gh/joblib/loky)
[![DOI](https://zenodo.org/badge/48578152.svg)](https://zenodo.org/badge/latestdoi/48578152)


### Goal

The aim of this project is to provide a robust, cross-platform and
cross-version implementation of the `ProcessPoolExecutor` class of
`concurrent.futures`. It notably features:

  * __Consistent and robust spawn behavior__: All processes are started
    using fork + exec on POSIX systems. This ensures safer interactions with
    third party libraries. On the contrary, `multiprocessing.Pool` uses
    fork without exec by default, causing third party runtimes to crash
    (e.g. OpenMP, macOS Accelerate...).

  * __Reusable executor__: strategy to avoid re-spawning a complete
    executor every time. A singleton executor instance can be reused (and
    dynamically resized if necessary) across consecutive calls to limit
    spawning and shutdown overhead. The worker processes can be shutdown
    automatically after a configurable idling timeout to free system
    resources.

  * __Transparent cloudpickle integration__: to call interactively
    defined functions and lambda expressions in parallel. It is also
    possible to register a custom pickler implementation to handle
    inter-process communications.

  * __No need for ``if __name__ == "__main__":`` in scripts__: thanks
    to the use of ``cloudpickle`` to call functions defined in the
    ``__main__`` module, it is not required to protect the code calling
    parallel functions under Windows.

  * __Deadlock free implementation__: one of the major concern in
    standard `multiprocessing` and `concurrent.futures` modules is the
    ability of the `Pool/Executor` to handle crashes of worker
    processes. This library intends to fix those possible deadlocks and
    send back meaningful errors. Note that the implementation of
    `concurrent.futures.ProcessPoolExecutor` that comes with Python 3.7+
    is as robust as the executor from loky but the latter also works for
    older versions of Python.


### Installation

The recommended way to install `loky` is with `pip`,
```bash
pip install loky
```

`loky` can also be installed from sources using
```bash
git clone https://github.com/joblib/loky
cd loky
python setup.py install
```

Note that `loky` has an optional dependency on [`psutil`][1] to allow early
memory leak detections.

### Usage

The basic usage of `loky` relies on the `get_reusable_executor`, which
internally manages a custom `ProcessPoolExecutor` object, which is reused or
re-spawned depending on the context.

```python
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
```

For more advance usage, see our
[documentation](https://loky.readthedocs.io/en/stable/)

### Workflow to contribute

To contribute to **loky**, first create an account on
[github](http://github.com/). Once this is done, fork the
[loky repository](http://github.com/loky/loky) to have your own repository,
clone it using 'git clone' on the computers where you want to work. Make your
changes in your clone, push them to your github account, test them on several
computers, and when you are happy with them, send a pull request to the main
repository.

### Running the test suite

To run the test suite, you need the `pytest` (version >= 3) and `psutil`
modules. From the root of the project, run the test suite using:

```sh
    pip install -e .
    pytest .
```

### Why was the project named `loky`?

While developping `loky`, we had some bad experiences trying to debug deadlocks
when using `multiprocessing.Pool` and `concurrent.futures.ProcessPoolExecutor`,
especially when calling functions with non-picklable arguments or returned
values at the beginning of the project. When we had to chose a name, we had
dealt with so many deadlocks that we wanted some kind of invocation to repel
them! Hence `loky`: a mix of a god, locks and the `y` that make it somehow
cooler and nicer : (and also less likely to result in name conflict in google
results ^^).

Fixes to avoid those deadlocks in `concurrent.futures` were also contributed
upstream in Python 3.7+, as a less mystical way to repel the deadlocks :D

### Acknowledgement

This work is supported by the Center for Data Science, funded by the IDEX
Paris-Saclay, ANR-11-IDEX-0003-02


[1]: https://github.com/giampaolo/psutil
