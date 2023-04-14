import os
import sys
import msvcrt
import _winapi

from multiprocessing import util
from multiprocessing.context import set_spawning_popen
from multiprocessing.popen_spawn_win32 import _close_handles
from multiprocessing.popen_spawn_win32 import Popen as _Popen

from . import reduction, spawn


__all__ = ["Popen"]

POPEN_FLAG = 0
if spawn.OPEN_CONSOLE_FOR_SUBPROCESSES:
    POPEN_FLAG = _winapi.CREATE_NEW_CONSOLE


#
#
#


def _path_eq(p1, p2):
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)


WINENV = hasattr(sys, "_base_executable") and not _path_eq(
    sys.executable, sys._base_executable
)


def _close_handles(*handles):
    for handle in handles:
        _winapi.CloseHandle(handle)


#
# We define a Popen class similar to the one from subprocess, but
# whose constructor takes a process object as its argument.
#


class Popen(_Popen):
    """
    Start a subprocess to run the code of a process object.

    We differ from cpython implementation with the way we handle environment
    variables, in order to be able to modify then in the child processes before
    importing any library, in order to control the number of threads in C-level
    threadpools.

    We also use the loky preparation data, in particular to handle main_module
    inits and the loky resource tracker.
    """

    method = "loky"

    def __init__(self, process_obj):
        prep_data = spawn.get_preparation_data(
            process_obj._name, getattr(process_obj, "init_main_module", True)
        )

        # read end of pipe will be duplicated by the child process
        # -- see spawn_main() in spawn.py.
        #
        # bpo-33929: Previously, the read end of pipe was "stolen" by the child
        # process, but it leaked a handle if the child process had been
        # terminated before it could steal the handle from the parent process.
        rhandle, whandle = _winapi.CreatePipe(None, 0)
        wfd = msvcrt.open_osfhandle(whandle, 0)

        cmd = spawn.get_command_line(
            pipe_handle=rhandle, parent_pid=os.getpid()
        )
        python_exe = cmd[0]
        cmd = " ".join(f'"{x}"' for x in cmd)

        # copy the environment variables to set in the child process
        child_env = {**os.environ, **process_obj.env}

        # bpo-35797: When running in a venv, we bypass the redirect
        # executor and launch our base Python.
        if WINENV and _path_eq(python_exe, sys.executable):
            cmd[0] = python_exe = sys._base_executable
            child_env["__PYVENV_LAUNCHER__"] = sys.executable


        cmd = " ".join(f'"{x}"' for x in cmd)

        with open(wfd, "wb") as to_child:
            # start process
            try:
                hp, ht, pid, _ = _winapi.CreateProcess(
                    python_exe,
                    cmd,
                    None,
                    None,
                    False,
                    0,
                    child_env,
                    None,
                    None,
                )
                _winapi.CloseHandle(ht)
            except BaseException:
                _winapi.CloseHandle(rhandle)
                raise

            # set attributes of self
            self.pid = pid
            self.returncode = None
            self._handle = hp
            self.sentinel = int(hp)
            self.finalizer = util.Finalize(
                self, _close_handles, (self.sentinel, int(rhandle))
            )

            # send information to child
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, to_child)
                reduction.dump(process_obj, to_child)
            finally:
                set_spawning_popen(None)
