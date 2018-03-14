# Reusable Process Pool Executor  [![Build Status](https://travis-ci.org/tomMoral/loky.svg?branch=master)](https://travis-ci.org/tomMoral/loky) [![Build status](https://ci.appveyor.com/api/projects/status/oifqilb5sb0p7fdp/branch/master?svg=true)](https://ci.appveyor.com/project/tomMoral/loky/branch/master) [![codecov](https://codecov.io/gh/tomMoral/loky/branch/master/graph/badge.svg)](https://codecov.io/gh/tomMoral/loky)


### Goal 

The aim of this project is to provide a robust, cross-platform and
cross-version implementation of the `ProcessPoolExecutor` class of
`concurrent.futures`. It notably features:

  * __Deadlock free implementation__: one of the major concern in
    standard `multiprocessing` and `concurrent.futures` libraries is the
    ability of the `Pool/Executor` to handle crashes of worker
    processes. This library intends to fix those possible deadlocks and
    send back meaningful errors.

  * __Consistent spawn behavior__: All processes are started using
    fork/exec on POSIX systems. This ensures safer interactions with
    third party libraries.

  * __Reusable executor__: strategy to avoid respawning a complete
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

### Usage

```python
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
```

### Workflow to contribute

To contribute to **loky**, first create an account on [github](http://github.com/).
Once this is done, fork the [loky repository](http://github.com/loky/loky) to
have your own repository, clone it using 'git clone' on the computers where you
want to work. Make your changes in your clone, push them to your github account,
test them on several computers, and when you are happy with them, send a pull
request to the main repository.

### Running the test suite

To run the test suite, you need the `pytest` (version >= 3) and `psutil`
modules. Run the test suite using:

```sh
    pip install -e .
    pytest .
```

from the root of the project.

### Acknowledgement

This work is supported by the Center for Data Science, funded by the IDEX
Paris-Saclay, ANR-11-IDEX-0003-02
