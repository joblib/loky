### 3.5.6 - 2025-08-27

- Fix ``resource_tracker`` compatibility with python 3.13.7+. (#461)

### 3.5.5 - 2025-05-22

- Fix ``resource_tracker`` teardown check for earlier python version. (#455)


### 3.5.3 - 2025-04-29

- Fix ``call_queue`` size in the ``_ReusableExecutor``, which could be too
  small when ``max_workers`` is larger than ``cpu_count()``. (#452)

### 3.5.2 - 2025-04-22

- Fix ``resource_tracker`` teardown to accommodate with newer version of
  Python (3.12.10+, 3.13.3+, 3.14+). (#450)

### 3.5.1 - 2025-03-18

- Fix a regression to support Python 3.10 (and Python 3.14 dev) by passing
  version-specific `allow_vfork` and `pgid_to_set` arguments to
  `_posixsubprocess.fork_exec`. (#445)

### 3.5.0 - 2025-03-14

- Avoid raising `DeprecationWarning` related to `os.fork` when running in a
  natively multi-threaded process. (#429).

- Fix a crash when calling commands that access `stdin` via `subprocess.run` in
  worker processes on POSIX systems. (#429).

- Automatically call `faulthandler.enable()` when starting loky worker
  processes to report more informative information (post-mortem Python
  tracebacks in particular) on worker crashs. (#419).

- Fix a random deadlock caused by a race condition at executor shutdown that
  was observed on Linux and Windows. (#438)

- Fix detection of the number of physical cores in
  `cpu_count(only_physical_cores=True)` on some Linux systems and recent
  Windows versions. (#425)

- Drop support for Python 3.7 and Python 3.8. (#409)

- Drop support for PyPy. (#427)

### 3.4.1 - 2023-06-29

- Fix compatibility with python3.7, which does not define
  a `_MAX_WINDOWS_WORKERS` constant. (#408)

### 3.4.0 - 2023-04-14

- Fix exception `__cause__` not being propagated with
  `tblib.pickling_support.install()`. (#255).

- Fix handling of CPU affinity  by using `psutil`'s `cpu_affinity` on platforms
  that do not implement `os.sched_getaffinity`, such as PyPy. (#381).

- Make the executor's gc process more thread-safe, in particular for PyPy,
  where the gc calls can be run in any thread. (#384).

- Fix crash when using `max_workers > 61` on Windows. Loky will no longer
  attempt to use more than 61 workers on that platform (or 60 depending on the
  Python version). (#390).

- Fix loky compat with python 3.11 for nested calls. (#394).

- Adapt the cooldown strategy when shutingdown an executor with full
  `call_queue`. This should accelerate the time taken to shutdown
  in general, in particular on overloaded machines. (#399).

### 3.3.0 - 2022-09-15

- Fix worker management logic in `get_reusable_executor` to ensure
  the number of started worker process actually correspond to `max_workers`
  when existing process concurrently time out (#370).

### 3.2.0 - 2022-09-14

- Fix leaked processes and deadlock when the Python interpreter exits
  after a using nested calls to `get_reusable_executor` (#363).

- Fix an exception in the SemLock finalizer when the semaphore has been
  concurrently unlinked (#366).

### 3.1.0 - 2022-02-22

- Fix loky.cpu_count() to properly detect the number of allowed CPUs based on
  the /sys/fs/cgroup/cpu.max file on newest Linux versions with cgroup v2.
  Fall-back to the /sys/fs/cgroup/cpu/cpu.cfs_quota_us file to keep on
  supporting Linux versions that use cgroup v1 (#355 and #358).

- Fix an exception that could be raised in an auxiliary thread when
  garbage collecting an executor instance when shutting down the
  the Python interpreter (#311).

- Make `shutdown(kill_workers=True)` consistently use the SIGKILL
  signal on POSIX. Previously a mix of SIGKILL and SIGTERM was issued
  and could deadlock the shutdown process (#348 and #357).

- Big code clean-up to drop support for older Python versions.
  Python 3.7 or later is now required. (#304)

### 3.0.0 - 2021-09-10

- Avoid a NameError when calling the `exit` builtin on Windows when
  loky is executed as part of a frozen Python binary. (#290)

- Make it possible to automatically trace workers when profiling with
  VizTracer (#299).

### 2.9.0 - 2020-10-02

- Fix a side-effect bug in the registration of custom reducers the loky
  subclass of `cloudpickle.CloudPickler` with cloudpickle 1.6.0. (#272).

- Fix support for Python 3.9 and test against python-nightly from now on
  (#250).

- Add a parameter to ``cpu_count``, ``only_physical_cores``, to return the
  number of physical cores instead of the number of logical cores (#271).

- Fix thread-safety issues when iterating over the list of processes
  (Dictionary changed sized during iteration) (#263).

### 2.8.0 - 2020-05-14

- Internal refactoring: add private factory class method to
  ``_ReusablePoolExecutor`` to ease extensibility in joblib (#253).

### 2.7.0 - 2020-04-30

- Increase the residual memory increase threshold  (100MB -> 300MB) used by
  loky for memory leak detection (causing loky workers to be
  shutdown/restarted), in order to reduce the amount of false positives (#238).

- In Python 3.8, loky processes now inherit multiprocessing's
  ``resource_tracker`` created from their parent. As a consequence, no spurious
  ``resource_tracker`` warnings are emitted when loky workers manipulate
  ``shared_memory`` objects (#242).
  Note that loky still needs to use its own resource tracker instance to manage
  resources that require the reference counting logic such as joblib temporary
  memory mapped files for now.

- The ``resource_tracker`` now comes with built-in support for tracking files
  in all OSes.  In addition, Python processes can now signal they do not need a
  shared resource anymore by using the
  ``resource_tracker.maybe_unlink(resource_name, resource_type)`` method.  After
  all processes having access to the said resource have called this method, the
  ``resource_tracker`` will proceed to unlink the resource. Previously, resource
  unlinking by the ``resource_tracker`` was only done for leaked resources at
  interpreter exit (#228).
- Fix `shutdown(wait=False)` that was potentially leading to deadlocks and froze
  interpreters (#246).
- Backport `ExecutorManagerThread` from cpython to refactor
  `_queue_management_thread` and ease maintenance (#246).

### 2.6.0 - 2019-09-18

- Copy the environment variables in the child process for ``LokyProcess``. Also
  add a ``env`` argument in ``LokyProcess``, ``ProcessPoolExecutor`` and
  ``get_reusable_executor`` to over-write consistently some environment variable
  in the child process. This allows setting env variables before loading any
  module.
  Note: this feature is unreliable on Windows with Python < 3.6. (#217)

- Fix a bug making all loky workers crash on Windows for Python>3.7 when using
  a virtual environment (#216).

### 2.5.1 - 2019-06-11 - Bugfix release

- Fix a bug of the ``resource_tracker``  that could create unlimited freeze on
  Windows (#212)


### 2.5.0 - 2019-06-07

- Backport ResourceTracker from Python 3.8 concurrent.futures and fix
  tracker pid issue (#204 and #202).

- Fix bug when pickling function with kw-only argument (#264).

- Fix bug in `pickler.dispatch_table` handling that could cause a crash
  with the new cloudpickle fast pickler under Python 3.8 (#203).

- Fix a race condition that could cause a deadlock with PyPy (#191).


### 2.4.2 - 2018-11-06 - Bugfix release

- Fixed loky pickler in workers. (#184)

- Fixed compat with python2.7 in semaphore tracker. (#186)

### 2.4.1 - 2018-11-02 - Bugfix release

- Fixed a bug when setting the cause of an exception without message
  under Python 2.7 (#179).

### 2.4.0 - 2018-11-01 - Release highlights

- Default serialization is now done with `cloudpickle`. (#178)

- The base `Pickler` in `loky` can now be changed through the `LOKY_PICKLER`
  variable or programmatically with `set_loky_pickler`. (#171)

- Improve reporting of causes in python2.7 (#174)

- Improve workers crash reporting by displaying the exitcodes of
  workers in `TerminatedWorkerError` (#173)

- Add a `wrap_non_picklable_objects` decorator in `loky` to make it
  easy to fix serialization failure for nested functions defined in
  the `__main__` module. (#171)

### 2.3.1 - 2018-09-13 - Bug fix release

- Improve error reporting when a worker process is terminated abruptly
  (#169).

- Remove spurious debug output.

### 2.3.0 - 2018-09-05 - Release highlights

- Add support for PyPy3.

- `loky.cpu_count()` is now upper-bounded by the value of the
  `LOKY_MAX_CPU_COUNT` environment variable (when defined).

- Fix issue #165 to make `loky.cpu_count()` return an integer under
  Python 2.7 with fractional docker CPU usage quotas.


### 2.2.2 - 2018-08-30 - Bug fix release

- Add a `set_start_method` function in `loky.backend.context`. Note
  that now, `loky` does not respect the start method set using
  `multiprocessing.set_start_method` anymore. It is thus mandatory
  to use the `loky` function to have the correct behavior.


### 2.2.1 - 2018-08-27 - Bug fix release

- Fix pickling logic in loky. Now the serialization is consistent
  between initializer and tasks. Also fixes the logic behind the
  environment variable `LOKY_PICKLER`.

- Improve reporting for tasks unpickling errors.

- Fix deadlock when large objects are sent to workers.

- Fix context and `start_method` logic for loky contexts.


### 2.2.0 - 2018-08-01 - Release Highlights

- Add a protection against memory-leaks for long running worker
  processes: if the memory usage has increased by more than 100 MB
  (after a garbage collection), the worker is automatically restarted
  before accepting new tasks. This protection is only active when psutil
  is installed.

- psutil is now a soft-dependency of loky: it is used to recursively
  terminate children processes when available but there is a fallback
  for windows and all unices with pgrep installed otherwise.

### 2.1.4 - 2018-06-29 - Bug fix release

- Fix win32 failure to kill worker process with taskkill returning 255
- Fix all error at pickle raise PicklingError
- Add missing license file

### 2.1.3 - 2018-06-20 - Release Highlights

- Fix bad interaction between `max_workers=None` and `reuse='auto'` (#132).
- Add initializer for `get_reusable_executor` (#134)

### 2.1.2 - 2018-06-04 - Release Highligths

- Fix terminate for nested processes
- Fix support for windows freezed application
- Fix some internal API inconsistencies

### 2.1.1 - 2018-04-13 - Bug fix release

- Fix interpreter shutdown
- Fix queue size in reusable executor
- Accelerate executor shutdown

### 2.1.0 - 2018-04-11 - Release Highlights

- Add documentation, accessible on http://loky.readthedocs.io
- Fixed a thread-safety issue when iterating over the dict of processes in
  `ReusablePoolExecutor._resize` and `ProcessPoolExecutor.shutdown`.

### 2.0 - 2018-03-15 - Release highlights

- Add `loky.cpu_count` that returns the number of CPUs the current process can
  use. This value is now used by default by `ProcessPoolExecutor`.  (#114).
- Add `__version__` field in the `loky` module (#110).
- Fix multiple instabilities and simplify the inner mechanisms (#105, #107).

### 1.2.2 - 2018-02-19 - Bug fix release

- Fix handling of Full Queue exception at shutdown time. (#107).

### 1.2.1 - 2017-10-13 - Bug fix release

- Fix a potential deadlock when shutting down a past reusable executor instance
  with to create a new instance with different parameters. (#101, #102)

### 1.2 - 2017-09-12 - Release highlights

- Rename `loky.process_executor.BrokenExecutor` as `loky.BrokenProcessPool` and
  subclass `concurrent.futures.process.BrokenProcessPool` when available to make
  except statements forward compatible.

### 1.1.4 - 2017-08-08 - Bug fix release

- Fix crash for 64-bit Python under Windows.
- Fix race condition in Queue Manager thread under Python 3.4+

### 1.1.2 - 2017-07-03 - Bug fix release

- Fix shutdown with exit at pickle deadlock
- Fix LokyProcess spawning forcing context to be loky in child (necessary for joblib)

### 1.1.1 - 2017-07-28 - Bug fix release

- Fix default backend for ReusablePoolExecutor

### 1.1 - 2017-07-28 - Release higlights

- Rename `loky` backend to `loky_init_main`.
- New `loky` backend which makes it possible to use `loky` with no `if __name__ == '__main__':` safeguard.
- Change the default backend to `loky`.
- Fix deadlocks on `PocessPoolExecutor.shutdown` (#71, #75)


### 1.0 - 2017-06-20 - Release highlights

- Make `ProcessPoolExecutor` use a spawn-based start method by default. The
  `fork`-based start method is not longer officially supported.
- Fixed a race condition at executor shutdown


### 0.3 - 2017-06-02 - Release highlights

- Basic handling of nested parallel calls up to recursion depth 3 (by default, can be changd by setting LOKY_MAX_DEPTH)
- Various internal code clean-up and test improvments


### 0.2 - 2017-03-24 - Release highlights

- Customizable serialization (#46)
- Add support for calling dynamically defined function when cloudpickle is available (#47)
- Fix resizing of the executor (#51)
- Various rare race condition fixes
