# flake8: noqa: F401
import sys

if sys.platform == "win32":
    from multiprocessing.popen_spawn_win32 import Popen
    from multiprocessing.connection import wait
    import _winapi
