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

