from time import sleep, time
from itertools import repeat

from ._long_import import test_import_length
from .sleep_and_return import sleep_and_return

HEADER = ["Name", "all", "start", "cold", "warm", "start+cold", "map chunk",
          "map sleep", "map id", "down"]

MAP_N_ITER = 1000
MAP_DELAY = 0.001
MAP_CHUNSIZE = 100


def test_executor(get_executor, **kwargs):
    t_start = time()
    executor = get_executor(**kwargs)
    t1 = time()

    executor.submit(test_import_length).result()
    t11 = time()

    executor.submit(test_import_length).result()
    t12 = time()

    for _ in executor.map(id, repeat(0, MAP_N_ITER), chunksize=MAP_CHUNSIZE):
        pass
    t2 = time()

    for _ in executor.map(sleep_and_return, repeat(MAP_DELAY, MAP_N_ITER)):
        pass
    t3 = time()

    for _ in executor.map(id, repeat(0, MAP_N_ITER)):
        pass
    t4 = time()

    executor.shutdown(wait=True)
    t5 = time()
    return [t5-t_start, t1-t_start, t11-t1, t12-t11, t11-t_start, t2-t12,
            t3-t2, t4-t3, t5-t4]


def test_pool(get_pool, **kwargs):
    t_start = time()
    pool = get_pool(**kwargs)
    t1 = time()

    pool.apply(test_import_length)
    t11 = time()

    pool.apply(test_import_length)
    t12 = time()

    for _ in pool.map(id, repeat(0, MAP_N_ITER), chunksize=MAP_CHUNSIZE):
        pass
    t2 = time()

    for _ in pool.map(sleep, repeat(MAP_DELAY, MAP_N_ITER), chunksize=1):
        pass
    t3 = time()

    for _ in pool.map(id, repeat(0, MAP_N_ITER), chunksize=1):
        pass
    t4 = time()

    pool.terminate()
    pool.join()
    t5 = time()

    return [t5-t_start, t1-t_start, t11-t1, t12-t11, t11-t_start, t2-t12,
            t3-t2, t4-t3, t5-t4]


def test_small_call(get_executor, delay=.001, terminate=False, **kwargs):
    t_start = time()
    executor = get_executor(**kwargs)
    t1 = time()

    for _ in executor.map(sleep_and_return, repeat(delay, MAP_N_ITER)):
        pass
    t2 = time()

    if terminate:
        executor.terminate()
        executor.join()

    return [t2-t_start, t1-t_start, t2-t1]
