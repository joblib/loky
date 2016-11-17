import sys
import multiprocessing as mp

from loky import process_executor
from ._executor_mixin import ExecutorMixin


if sys.version_info[:2] > (3, 3) and sys.platform != "win32":
    class ProcessPoolForkserverMixin(ExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = mp.get_context('forkserver')

    from ._test_process_executor import ExecutorShutdownTest

    class TestsProcessPoolForkserverShutdown(ProcessPoolForkserverMixin,
                                             ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolForkserverWait(ProcessPoolForkserverMixin,
                                         WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolForkserverAsCompleted(ProcessPoolForkserverMixin,
                                                AsCompletedTests):
        pass

    from ._test_process_executor import ExecutorTest

    class TestsProcessPoolForkserverExecutor(ProcessPoolForkserverMixin,
                                             ExecutorTest):
        pass

    from ._test_process_executor import ExecutorTest
