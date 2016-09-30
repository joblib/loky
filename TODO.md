### TODO list


### ProcessPoolExecutor

- [x] cancel_on_shutdown
- [x] robustification dans backend
- [x] idle workers should shutdown while executor should restart missing worker on submit
- [x] modification parameters ==> recreate pool
- [x] debug lock on start executor
- [x] support python2.7 (kwargs.pop ? ) regardé concurrent future package
- [ ] cutsomizable pickler for the queues in ProcessPoolExecutor
- [ ] remove appveyor logs

#### Test

- [x] Test for errors and crashes in job dispatch by the apply async callback
- [x] Configure Appveyor to run the test under windows (see joblib one)
- [x] Add travis to run the test under linux/ mac
- [x] Unskip numpy freeze test once Spawn Processes are implemented
- [ ] Test the callbacks call with ProcessPoolExecutor
- [ ] Unit test semaphore and fd managment
- [ ] Test spaCy openMP freeze with cached nlp model on travis/Appveyor
      https://gist.github.com/ogrisel/a461d68c8e5b0a5aabb496a726b52f8b


#### Python upstream

- [ ] Identify and contribute to pool stability in upstream multiprocessing in pydev
- [x] See if Rpool pass Pool test suite from cpython

#### Spawn Processes

- [x] Create process with subprocess.Popen and os.pipe for posix
- [x] Test processes with cpython test suite
- [x] Pass test rpool with such processes
- [ ] Benchmark

#### Joblib

- [ ] Implement new backend based on loky
- [ ] Make new backend auto mmemap large numpy
