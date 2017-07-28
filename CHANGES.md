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

