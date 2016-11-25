import sys
import multiprocessing as mp

from loky import process_executor
from ._executor_mixin import ExecutorMixin


if sys.version_info[:2] > (3, 3) and sys.platform != "win32":
    class ProcessPoolForkMixin(ExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = mp.get_context('fork')
        # Increase the minimal worker timeout for OSX with fork as some weird
        # behaviors occurs in with this case. This should be investigated but
        # it is not a priority. With a timeout set at .01, some worker does not
        # start properly in test_worker_timeout
        if sys.platform == "darwin":
                min_worker_timeout = .1

    from ._test_process_executor import ExecutorShutdownTest

    class TestsProcessPoolForkShutdown(ProcessPoolForkMixin,
                                       ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolForkWait(ProcessPoolForkMixin, WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolForkAsCompleted(ProcessPoolForkMixin,
                                          AsCompletedTests):
        pass

    from ._test_process_executor import ExecutorTest

    class TestsProcessPoolForkExecutor(ProcessPoolForkMixin, ExecutorTest):
        pass

    from ._test_process_executor import ExecutorTest
