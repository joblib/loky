import gc
import os
import sys
import threading
import warnings
import weakref
from collections import Counter

import pytest

from loky import ProcessPoolExecutor, get_reusable_executor
from loky.backend import get_context
from loky.process_executor import ShutdownExecutorError, TerminatedWorkerError

from .utils import check_python_subprocess_call

CONTEXTS = ["loky", "spawn"]
if sys.platform != "win32":
    CONTEXTS.append("forkserver")


def task_pid(value=None):
    return os.getpid()


def fail_task():
    raise ValueError(str(os.getpid()))


_worker_event = None


def initialized_task():
    _worker_event.wait(20)
    return os.environ["LOKY_RECYCLE_TEST"], os.getpid()


def initialize_worker(value, event):
    global _worker_event
    _worker_event = event
    os.environ["LOKY_RECYCLE_TEST"] = value


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_value(limit):
    with pytest.raises(ValueError, match="max_tasks_per_child"):
        ProcessPoolExecutor(max_tasks_per_child=limit)


@pytest.mark.parametrize("limit", [1.5, "2", object()])
def test_invalid_limit_type(limit):
    with pytest.raises(
        TypeError, match="max_tasks_per_child must be an integer"
    ):
        ProcessPoolExecutor(max_tasks_per_child=limit)


@pytest.mark.skipif(sys.platform == "win32", reason="fork is POSIX only")
def test_fork_is_rejected():
    with pytest.warns(UserWarning, match="fork"):
        context = get_context("fork")
    with pytest.raises(ValueError, match="fork"):
        ProcessPoolExecutor(context=context, max_tasks_per_child=1)


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("limit", [1, 2])
def test_repeated_recycling(context, limit):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with ProcessPoolExecutor(
            max_workers=2,
            context=get_context(context),
            max_tasks_per_child=limit,
        ) as executor:
            futures = [executor.submit(task_pid) for _ in range(17)]
            pids = [future.result(30) for future in futures]
            assert max(Counter(pids).values()) <= limit
            assert len(set(pids)) >= (len(pids) + limit - 1) // limit
            assert executor.submit(task_pid).result(30)
    assert not [w for w in recorded if "worker" in str(w.message).lower()]


@pytest.mark.parametrize("context", CONTEXTS)
def test_exceptions_count_towards_limit(context):
    with ProcessPoolExecutor(
        max_workers=1, context=get_context(context), max_tasks_per_child=2
    ) as executor:
        first_pid = executor.submit(task_pid).result(30)
        with pytest.raises(ValueError) as error:
            executor.submit(fail_task).result(30)
        assert str(first_pid) == str(error.value)
        assert executor.submit(task_pid).result(30) != first_pid


def test_map_counts_chunks():
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=2) as executor:
        pids = list(executor.map(task_pid, range(15), chunksize=3, timeout=30))
    assert pids[0] == pids[5]
    assert pids[6] == pids[11]
    assert pids[12] == pids[14]
    assert len({pids[0], pids[6], pids[12]}) == 3


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("mode", ["shutdown_wait", "shutdown_nowait", "gc"])
def test_pending_tasks_survive_executor_lifecycle(mode, context):
    context = get_context(context)
    event = context.Event()
    executor = ProcessPoolExecutor(
        max_workers=1,
        context=context,
        max_tasks_per_child=1,
        initializer=initialize_worker,
        initargs=("initialized", event),
    )
    futures = [executor.submit(initialized_task) for _ in range(5)]
    manager = executor._executor_manager_thread
    executor_ref = weakref.ref(executor)
    if mode == "shutdown_nowait":
        executor.shutdown(wait=False)
    elif mode == "gc":
        del executor
        gc.collect()
        assert executor_ref() is None
    event.set()
    if mode == "shutdown_wait":
        executor.shutdown()
    results = [future.result(30) for future in futures]
    assert all(value == "initialized" for value, _ in results)
    assert len({pid for _, pid in results}) == len(results)
    manager.join(30)
    assert not manager.is_alive()
    assert not manager.processes


def test_idle_recycling_and_executor_reuse():
    executor = get_reusable_executor(max_workers=1, max_tasks_per_child=1)
    try:
        first_pid = executor.submit(task_pid).result(30)
        reused = get_reusable_executor(max_workers=1, max_tasks_per_child=1)
        assert reused is executor
        assert reused.submit(task_pid).result(30) != first_pid
        changed = get_reusable_executor(max_workers=1, max_tasks_per_child=2)
        assert changed is not executor
        first_pid = changed.submit(task_pid).result(30)
        assert changed.submit(task_pid).result(30) == first_pid
        assert changed.submit(task_pid).result(30) != first_pid
        resized = get_reusable_executor(max_workers=2, max_tasks_per_child=2)
        assert resized is changed
        assert resized._max_workers == 2
        assert list(resized.map(task_pid, range(8), timeout=30))
    finally:
        get_reusable_executor(reuse=True).shutdown()


def test_default_keeps_worker():
    with ProcessPoolExecutor(max_workers=1) as executor:
        assert len(set(executor.map(task_pid, range(5), timeout=30))) == 1


def test_crash_is_not_recycling():
    executor = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)
    try:
        with pytest.raises(TerminatedWorkerError):
            executor.submit(os._exit, 1).result(30)
    finally:
        executor.shutdown()


class Unpicklable:
    def __reduce__(self):
        raise ValueError(str(os.getpid()))


def unpicklable_result():
    return Unpicklable()


def test_serialization_failures_count_only_executed_calls():
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=2) as executor:
        first_pid = executor.submit(task_pid).result(30)
        with pytest.raises(Exception):
            executor.submit(task_pid, Unpicklable()).result(30)
        # The argument failed in the parent; the worker still has one call left.
        with pytest.raises(ValueError) as error:
            executor.submit(unpicklable_result).result(30)
        assert str(error.value) == str(first_pid)
        # Producing an unpicklable result does consume that second call.
        assert executor.submit(task_pid).result(30) != first_pid


def test_cancelled_tasks_do_not_count():
    context = get_context("loky")
    event = context.Event()
    with ProcessPoolExecutor(
        max_workers=1,
        max_tasks_per_child=3,
        initializer=initialize_worker,
        initargs=("initialized", event),
    ) as executor:
        futures = [executor.submit(initialized_task) for _ in range(20)]
        cancelled = futures.pop()
        assert cancelled.cancel()
        event.set()
        pids = [future.result(30)[1] for future in futures]
        counts = list(Counter(pids).values())
        assert counts == [3] * 6 + [1]
        assert cancelled.cancelled()


def test_kill_workers_with_pending_tasks():
    context = get_context("loky")
    event = context.Event()
    executor = ProcessPoolExecutor(
        max_workers=1,
        max_tasks_per_child=1,
        initializer=initialize_worker,
        initargs=("initialized", event),
    )
    futures = [executor.submit(initialized_task) for _ in range(10)]
    manager = executor._executor_manager_thread
    executor.shutdown(kill_workers=True)
    assert not manager.is_alive()
    assert not manager.processes
    for future in futures:
        with pytest.raises(ShutdownExecutorError):
            future.result(30)


def nested_task():
    inner = get_reusable_executor(max_workers=1)
    return inner.submit(task_pid).result(30)


def test_nested_executor_cleanup():
    psutil = pytest.importorskip("psutil")
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        descendants = [
            executor.submit(nested_task).result(30) for _ in range(3)
        ]
    assert all(not psutil.pid_exists(pid) for pid in descendants)


def test_interpreter_exit_drains_pending_tasks():
    check_python_subprocess_call(
        """
import atexit
from loky import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)
futures = [executor.submit(pow, 2, n) for n in range(6)]
def verify():
    assert [future.result(timeout=0) for future in futures] == [1, 2, 4, 8, 16, 32]
    print("all queued tasks completed")
atexit.register(verify)
""",
        stdout_regex="all queued tasks completed",
        timeout=40,
    )


def initialize_nested_cleanup(release, cleanup_started):
    from loky import process_executor

    initialize_worker("outer", release)
    original_exit = process_executor._python_exit

    def observed_exit():
        cleanup_started.set()
        original_exit()

    process_executor._python_exit = observed_exit


def start_unfinished_nested_task():
    inner = get_reusable_executor(
        max_workers=1,
        initializer=initialize_worker,
        initargs=("inner", _worker_event),
    )
    inner_pid = inner.submit(task_pid).result(30)
    inner.submit(initialized_task)
    return os.getpid(), inner_pid


def test_kill_workers_during_quota_nested_cleanup():
    psutil = pytest.importorskip("psutil")
    context = get_context("loky")
    release = context.Event()
    cleanup_started = context.Event()
    executor = ProcessPoolExecutor(
        max_workers=1,
        max_tasks_per_child=1,
        initializer=initialize_nested_cleanup,
        initargs=(release, cleanup_started),
    )
    shutdown_thread = None
    try:
        pids = executor.submit(start_unfinished_nested_task).result(30)
        assert cleanup_started.wait(30)
        shutdown_thread = threading.Thread(
            target=executor.shutdown, kwargs={"kill_workers": True}
        )
        shutdown_thread.start()
        shutdown_thread.join(5)
        assert not shutdown_thread.is_alive()
        assert all(not psutil.pid_exists(pid) for pid in pids)
    finally:
        # Bound cleanup if shutdown was blocked by the nested task. After
        # successful forced termination, the event's lock may be abandoned.
        if shutdown_thread is None or shutdown_thread.is_alive():
            release.set()
        if shutdown_thread is not None:
            shutdown_thread.join(30)
        executor.shutdown(kill_workers=True)


def hold_quota_cleanup(release, reached):
    from loky import process_executor

    original_exit = process_executor._python_exit

    def held_exit():
        reached.put(os.getpid())
        release.wait(30)
        original_exit()

    process_executor._python_exit = held_exit


def test_resize_after_all_workers_reach_quota():
    context = get_context("loky")
    release = context.Event()
    reached = context.Queue()
    executor = get_reusable_executor(
        max_workers=2,
        max_tasks_per_child=1,
        initializer=hold_quota_cleanup,
        initargs=(release, reached),
        reuse=False,
    )
    helper = None
    try:
        pids = list(executor.map(task_pid, range(2), timeout=30))
        assert len(set(pids)) == 2
        assert {reached.get(timeout=30), reached.get(timeout=30)} == set(pids)
        sentinel_sent = threading.Event()
        original_put = executor._call_queue.put

        def observe_put(item, *args, **kwargs):
            result = original_put(item, *args, **kwargs)
            if item is None:
                sentinel_sent.set()
            return result

        executor._call_queue.put = observe_put

        def unblock():
            sentinel_sent.wait(10)
            release.set()

        helper = threading.Thread(target=unblock)
        helper.start()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            resized = get_reusable_executor(
                max_workers=1,
                max_tasks_per_child=1,
                initializer=hold_quota_cleanup,
                initargs=(release, reached),
            )
            assert resized is executor
            assert executor.submit(task_pid).result(30)
            executor.shutdown()
        assert not recorded
    finally:
        release.set()
        if helper is not None:
            helper.join(30)
        executor.shutdown(kill_workers=True)
        reached.close()
        reached.join_thread()
