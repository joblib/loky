import signal
import multiprocessing as mp

from openmp_parallel_sum import parallel_sum


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Test compat openMP/multproc')
    parser.add_argument('--start', type=str, default='fork',
                        help='define start method tested with openMP')
    args = parser.parse_args()

    parallel_sum(10)

    mp.set_start_method(args.start)
    p = mp.Process(target=parallel_sum, args=(10,))

    def raise_timeout(a, b):
        print("TIMEOUT - parallel_sum could not complete in less than a sec")
        p.terminate()

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(1)
    p.start()
    p.join()
    print("done")
