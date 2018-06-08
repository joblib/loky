import time
import signal
import multiprocessing as mp

from parallel_sum import parallel_sum


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Test compat openMP/multproc')
    parser.add_argument('--start', type=str, default='fork',
                        help='define start method tested with openMP')
    args = parser.parse_args()

    t_start = time.time()
    parallel_sum(10)
    dt_parallel_sum = time.time() - t_start
    timeout = 3

    mp.set_start_method(args.start)
    p = mp.Process(target=parallel_sum, args=(10,))

    def raise_timeout(a, b):
        print("DEADLOCK - parallel_sum took {:.2f}ms in the main process but "
              "could not terminate in {}s in the subprocess. The computation "
              "are stucked because we used `fork` to start our new process, "
              "breaking the POSIX convention.".format(
                  dt_parallel_sum * 1000, timeout)
              )
        p.terminate()

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(timeout)
    p.start()
    p.join()
