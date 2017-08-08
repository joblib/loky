### 1.1.4 - 08/08/2017 - Bug fix release

- Fix crash for 64-bit Python under Windows.
- Fix race condition in Queue Manager thread under Python 3.4+

### 1.1.2 - 30/07/2017 - Bug fix release

- Fix shutdown with exit at pickle deadlock
- Fix LokyProcess spawning forcing context to be loky in child (necessary for joblib)

### 1.1.1 - 28/07/2017 - Bug fix release

- Fix default backend for ReusablePoolExecutor

### 1.1 - 28/07/2017 - Release higlights

- Rename `loky` backend to `loky_init_main`.
- New `loky` backend which makes it possible to use `loky` with no `if __name__ == '__main__':` safeguard.
- Change the default backend to `loky`.
- Fix deadlocks on `PocessPoolExecutor.shutdown` (#71, #75)


### 1.0 - 20/06/2017 - Release highlights

- Make `ProcessPoolExecutor` use a spawn-based start method by default. The
  `fork`-based start method is not longer officially supported.
- Fixed a race condition at executor shutdown


### 0.3 - 02/06/2017 - Release highlights

- Basic handling of nested parallel calls up to recursion depth 3 (by default, can be changd by setting LOKY_MAX_DEPTH)
- Various internal code clean-up and test improvments


### 0.2 - 24/03/2017 - Release highlights

- Customizable serialization (#46)
- Add support for calling dynamically defined function when cloudpickle is available (#47)
- Fix resizing of the executor (#51)
- Various rare race condition fixes

