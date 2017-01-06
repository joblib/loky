import numpy as np
import sys
from loky.reusable_executor import get_reusable_executor
from loky.process_executor import ProcessPoolExecutor as PPE
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

from benchmark.benchmark_helpers import test_executor, test_pool, HEADER,\
    test_small_call

run_exec = [
        ("loky", get_reusable_executor),
        ("concurrent", ProcessPoolExecutor)
]
if sys.version_info[:2] < (3, 3) or sys.platform == "win32":
    from multiprocessing import Pool
    run_pool = [
            ("fork", Pool)
    ]
else:
    from multiprocessing import get_context
    context_fork = get_context("fork")
    context_serv = get_context("forkserver")
    context_spawn = get_context("spawn")

    def PPEF(max_workers=1):
        e = PPE(max_workers=max_workers, context=context_fork)
        return e

    def PPEFS(max_workers=1):
        e = PPE(max_workers=max_workers, context=context_serv)
        return e
    run_exec += [
        ("executor fork", PPEF),
        ("executor serv", PPEFS)
    ]
    run_pool = [
        ("fork", context_fork.Pool),
        ("serv", context_serv.Pool),
        ("spawn", context_spawn.Pool)
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Benchmarking of the loky processes')
    parser.add_argument('-N', type=int, default=10,
                        help='number of run averaged to compute performances')
    parser.add_argument('--rep', action="store_true",
                        help='use the small repetitive task benchmark')
    parser.add_argument('--delay', type=float, default=.001,
                        help='delay used for the repetitive tasks')

    args = parser.parse_args()
    if not args.rep:
        N = args.N
        max_workers = 8
        results = defaultdict(lambda: [])
        M = len(run_pool) + len(run_exec)
        count = 0
        total = N*M
        for i in range(N):
            for n, e in run_exec:
                results[n] += [test_executor(e, max_workers=max_workers)]
                count += 1
                sys.stdout.write("\rbenchmark: {:7.2%}".format(count/total))
                sys.stdout.flush()
            for n, p in run_pool:
                results[n] += [test_pool(p, processes=max_workers)]
                count += 1
                sys.stdout.write("\rbenchmark: {:7.2%}".format(count/total))
                sys.stdout.flush()
        print("\rbenchmark: {:7}".format("done"))

        frmt = "{:>10}\t"+"{:10.6f}\t"*9
        frmt_h = "{:>10}\t"*10
        print(frmt_h.format(*HEADER))
        for n, _ in run_exec+run_pool:
            res = results[n]
            res = np.sort(res, axis=0)
            print(frmt.format(n, *np.mean(res[2:-2], axis=0)))
    else:

        def get_pool(max_workers=1):
            return context_fork.Pool(max_workers)

        res = np.array([test_small_call(
            get_reusable_executor, max_workers=10, delay=args.delay)
                         for _ in range(50)])
        res_pool = np.array([test_small_call(
            get_pool, max_workers=10, terminate=True, delay=args.delay)
             for _ in range(50)])

        res.sort(axis=0)
        res_pool.sort(axis=0)
        print("loky", ("{:.4f}\t"*3).format(*(res[10:-10].mean(axis=0))))
        print("pool", ("{:.4f}\t"*3).format(*(res_pool[10:-10].mean(axis=0))))
