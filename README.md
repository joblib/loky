# Reusable Process Pool Executor  [![Build Status](https://travis-ci.org/tomMoral/loky.svg?branch=master)](https://travis-ci.org/tomMoral/loky) [![Build status](https://ci.appveyor.com/api/projects/status/7jwt6ys4axq4feoj?svg=true)](https://ci.appveyor.com/project/tomMoral/rpool)

### Goal 
The aim of this project is to provide a robust, cross plateform and cross version implementation of the `ProcessPoolExecutor` of `concurrent.futures`.  
It features:

  * __Deadlock free implementation__: one of the major concern in standard `multiprocessing` and `concurrent.futures` libraries is the ability of the `Pool/Executor` to handle crashes. This library intends to fix those possible deadlocks and send back meaningful errors.
  * __Consistent spawn behavior__: All processes are started using fork/exec on POSIX systems. This ensures safer interactions with third party libraries.
  * __Reusable executor__: strategy to avoid respawning a complete executor every time. A singleton pool can be reused (and dynamically resized if necessary) accross calls. The workers can be shutdown automatically after timeout idle.

### Usage

```python
import os
from time import sleep
from loky.reusable_executor import get_reusable_executor


def say_hello(k):
    pid = os.getpid()
    print("hello from {} with arg {}".format(pid, k))
    sleep(.01)
    return pid

if __name__ == "__main__":
    # Create an executor with max_workers workers, that will shutdown after 2s
    max_workers = 4
    executor = get_reusable_executor(max_workers=max_workers, timeout=2)

    res = executor.submit(say_hello, 1)
    print("got result", res.result())

    results = executor.map(say_hello, range(50))
    n_workers = len(set(results))
    print("# used processes:", n_workers)
    assert n_workers == max_workers

    executor.shutdown(wait=True)

```

### Acknowledgement

This work is supported by the Center for Data Science, funded by the IDEX Paris-Saclay, ANR-11-IDEX-0003-02
