import sys
import multiprocessing as mp

from loky import process_executor
from ._executor_mixin import ExecutorMixin
from ._test_process_executor import exit_on_deadlock  # noqa


if (sys.version_info[:2] > (3, 3) and sys.platform != "win32") or (
        sys.version_info[:2] <= (3, 3)):

    if sys.version_info[:2] > (3, 3):
        class ProcessPoolLokyMixin(ExecutorMixin):
            # Makes sure that the context is defined
            from loky import backend
            executor_type = process_executor.ProcessPoolExecutor
            context = mp.get_context('loky')
    else:
        class ProcessPoolLokyMixin(ExecutorMixin):
            executor_type = process_executor.ProcessPoolExecutor
            from loky import backend
            context = backend

    from ._test_process_executor import ExecutorShutdownTest

    class TestsProcessPoolLokyShutdown(ProcessPoolLokyMixin,
                                       ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolLokyWait(ProcessPoolLokyMixin,
                                   WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolLokyAsCompleted(ProcessPoolLokyMixin,
                                          AsCompletedTests):
        pass

    from ._test_process_executor import ExecutorTest

    class TestsProcessPoolLokyExecutor(ProcessPoolLokyMixin,
                                       ExecutorTest):
        pass
