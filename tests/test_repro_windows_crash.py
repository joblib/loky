import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor


@pytest.mark.skipif(np is None, reason="requires numpy")
def test_numpy_dot_parent_and_child_no_freeze():
    """Test that no freeze happens in child process when numpy's thread
    pool is started in the parent.
    """
    a = np.random.randn(1000, 1000)
    np.dot(a, a)  # trigger the thread pool init in the parent process

    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.submit(np.dot, a, a).result()
