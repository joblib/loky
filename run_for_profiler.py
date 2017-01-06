from itertools import repeat
from loky.reusable_executor import get_reusable_executor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


N_ITER = 50000


def long_executor(get_executor, chunksize=1, **kwargs):
    with get_executor(**kwargs) as executor:
        for _ in executor.map(id, repeat(0, N_ITER),
                              chunksize=chunksize):
            pass


def long_pool(get_pool, chunksize=1, **kwargs):
    pool = get_pool(**kwargs)
    for _ in pool.map(id, repeat(0, N_ITER), chunksize=chunksize):
        pass
    pool.terminate()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiemnt')
    parser.add_argument('--run', type=str, default=None,
                        help='run loky or exec')
    parser.add_argument('--chunksize', type=int, default=1,
                        help='choose chunksize')

    args = parser.parse_args()
    max_workers = 8
    if args.run == "loky":
        long_executor(get_reusable_executor, max_workers=max_workers,
                      chunksize=args.chunksize)
    elif args.run == "pool":
        long_pool(Pool, processes=max_workers,
                  chunksize=args.chunksize)
    elif args.run == "ccr":
        long_executor(ProcessPoolExecutor, max_workers=max_workers,
                      chunksize=args.chunksize)
