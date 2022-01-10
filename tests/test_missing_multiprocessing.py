"""
Pyodide and other single-threaded Python builds will be missing the
_multiprocessing module. Test that loky still works in this environment.
"""

import os
import subprocess
import sys


def test_missing_multiprocessing():
    """
    Test that import loky works even if _multiprocessing is missing.

    pytest has already imported everything from loky. The most reasonable way
    to test importing joblib with modified environment is to invoke a separate
    Python process. This also ensures that we don't break other tests by
    importing a bad `_multiprocessing` module.
    """
    env = dict(os.environ)
    # For subprocess, use current sys.path with our custom version of
    # multiprocessing inserted.
    import loky
    from pathlib import Path
    missing_multiprocessing_path = str((Path(loky.__path__[0]) / "../tests/missing_multiprocessing").resolve())
    print(missing_multiprocessing_path)
    env["PYTHONPATH"] = ":".join(
        [missing_multiprocessing_path] + sys.path
    )
    subprocess.check_call([sys.executable, "-c", "import loky; import sys; print(sys.modules['_multiprocessing'])"], env=env)
