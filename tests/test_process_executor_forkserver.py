import sys

from loky import process_executor
from loky.backend import get_context

from ._executor_mixin import ExecutorMixin


if sys.platform != "win32" and not hasattr(sys, "pypy_version_info"):
    # XXX: the forkserver backend is broken with pypy3.
    from ._test_process_executor import (
        AsCompletedTests,
        ExecutorShutdownTest,
        ExecutorTest,
        WaitTests
    )

    class ProcessPoolForkserverMixin(ExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = get_context('forkserver')

    class TestsProcessPoolForkserverShutdown(ProcessPoolForkserverMixin,
                                             ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    class TestsProcessPoolForkserverWait(ProcessPoolForkserverMixin,
                                         WaitTests):
        pass

    class TestsProcessPoolForkserverAsCompleted(ProcessPoolForkserverMixin,
                                                AsCompletedTests):
        pass

    class TestsProcessPoolForkserverExecutor(ProcessPoolForkserverMixin,
                                             ExecutorTest):
        pass
