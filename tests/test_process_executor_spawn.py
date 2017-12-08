import sys
import pytest

from loky import process_executor
from loky.backend import get_context
from ._executor_mixin import ExecutorMixin


# ignore the worker timeout warnings for all tests in this class
pytestmark = pytest.mark.filterwarnings('ignore:A worker timeout')


if sys.version_info[:2] > (3, 3):
    class ProcessPoolSpawnMixin(ExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = get_context('spawn')

    from ._test_process_executor import ExecutorShutdownTest

    class TestsProcessPoolSpawnShutdown(ProcessPoolSpawnMixin,
                                        ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolSpawnWait(ProcessPoolSpawnMixin, WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolSpawnAsCompleted(ProcessPoolSpawnMixin,
                                           AsCompletedTests):
        pass

    from ._test_process_executor import ExecutorTest

    class TestsProcessPoolSpawnExecutor(ProcessPoolSpawnMixin, ExecutorTest):
        pass

    from ._test_process_executor import ExecutorTest
