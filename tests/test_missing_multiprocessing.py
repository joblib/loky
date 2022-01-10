"""
Pyodide and other single-threaded Python builds will be missing the
_multiprocessing module. Test that loky still works in this environment.
"""

import os
import pytest
import subprocess
import sys


def test_missing_multiprocessing(tmp_path):
    """
    Test that import loky works even if _multiprocessing is missing.

    pytest has already imported everything from loky. The most reasonable way
    to test importing joblib with modified environment is to invoke a separate
    Python process. This also ensures that we don't break other tests by
    importing a bad `_multiprocessing` module.
    """
    if sys.version_info[0] == 2:
        pytest.skip("pathlib is missing in Python 2")

    (tmp_path / "_multiprocessing.py").write_text('raise ImportError("No _multiprocessing module!")')
    env = dict(os.environ)
    # For subprocess, use current sys.path with our custom version of
    # multiprocessing inserted.
    env["PYTHONPATH"] = ":".join(
        [str(tmp_path)] + sys.path
    )
    subprocess.check_call([sys.executable, "-c", "import loky"], env=env)
