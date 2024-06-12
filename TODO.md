### TODO list


### ProcessPoolExecutor

- [x] cancel_on_shutdown
- [x] robustification dans backend
- [x] idle workers should shutdown while executor should restart missing worker on submit
- [x] modification parameters ==> recreate pool
- [x] debug lock on start executor
- [x] support python2.7 (kwargs.pop ? ) regardé concurrent future package
- [x] remove appveyor logs
- [ ] cutsomizable pickler for the queues in ProcessPoolExecutor
- [ ] cleanup compat files
- [ ] refactor thread to optimize communication (_feeder thread)  #12

#### Test

- [x] Test for errors and crashes in job dispatch by the apply async callback
- [x] Configure Appveyor to run the test under windows (see joblib one)
- [x] Add travis to run the test under linux/ mac
- [x] Unskip numpy freeze test once Spawn Processes are implemented
- [x] Test the callbacks call with ProcessPoolExecutor
- [ ] Unit test semaphore and fd managment
- [ ] Make cython model only built in test that highlight the OpenMP freeze with
      multiprocessing fork


#### Python upstream

- [x] See if Rpool pass Pool test suite from cpython
- [ ] Do not use the result queue to wakeup _queue_management
- [ ] Perfomance optimization for _queue_management_thread wakeup
- [ ] Add broken process pool detection thread (after #12 is solved)

#### Spawn Processes

- [x] Create process with subprocess.Popen and os.pipe for posix
- [x] Test processes with cpython test suite
- [x] Pass test rpool with such processes
- [-] automatic benchmark script launched in CI after test (#11)
- [ ] benchmark large messages (string)

#### Joblib

- [ ] Implement new backend based on loky
- [ ] Make new backend auto mmemap large numpy
