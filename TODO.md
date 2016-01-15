### TODO list

#### Test

- [ ] Test for errors and crashes in job dispatch by the apply async callback
- [ ] Configure Appveyor to run the test under windows (see joblib one)
- [ ] Add travis to run the test under linux/ mac
- [ ] Unskip numpy freeze test once Spawn Processes are implemented


#### Python upstream

- [ ] Identify and contribute to pool stability in upstream multiprocessing in pydev
- [ ] See if Rpool pass Pool test suite from cpython

#### Spawn Processes

- [ ] Create process with subprocess.Popen and os.pipe for posix
- [ ] Benchmark
- [ ] Use dill for serialization of callable
- [ ] Test processes with cpython test suite
- [ ] Pass test rpool with such processes

#### Joblib

- See later
