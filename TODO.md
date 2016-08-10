### TODO list


### ProcessPoolExecutor

- [x] cancel_on_shutdown
- [x] robustification dans backend
- [ ] idle workers should shutdown while executor should restart missing worker on submit
- [x] modification parameters ==> recreate pool
- [ ] cutsomizable queue in ProcessPoolExecutor
- [ ] support python2.7 (kwargs.pop ? ) regardé concurrent future package
- [ ] debug lock on start executor

#### Test

- [x] Test for errors and crashes in job dispatch by the apply async callback
- [x] Configure Appveyor to run the test under windows (see joblib one)
- [x] Add travis to run the test under linux/ mac
- [ ] Unskip numpy freeze test once Spawn Processes are implemented


#### Python upstream

- [ ] Identify and contribute to pool stability in upstream multiprocessing in pydev
- [x] See if Rpool pass Pool test suite from cpython

#### Spawn Processes

- [ ] Create process with subprocess.Popen and os.pipe for posix
- [ ] Benchmark
- [ ] Use dill for serialization of callable
- [ ] Test processes with cpython test suite
- [ ] Pass test rpool with such processes

#### Joblib

- See later
