import os
import sys
import signal
import pickle
from io import BytesIO

from . import reduction, spawn
from multiprocessing import util

if sys.version_info[:2] > (3, 3):
    from multiprocessing import context
    from .process import ExecContext
    context._concrete_contexts['exec'] = ExecContext()
elif sys.version_info[:2] < (3, 3):
    ProcessLookupError = OSError

if sys.platform != "win32":
    from . import semaphore_tracker

__all__ = ['Popen']

_spawning_popen = None


def is_spawning():
    return _spawning_popen is not None


def set_spawning_popen(popen):
    global _spawning_popen
    _spawning_popen = popen


def get_spawning_popen():
    global _spawning_popen
    return _spawning_popen

#
# Wrapper for an fd used while launching a process
#


class _DupFd(object):
    def __init__(self, fd):
        self.fd = reduction._mk_inheritable(fd)
    def detach(self):
        return self.fd

#
# Start child process using subprocess.Popen
#

class Popen(object):
    method = 'exec'
    DupFd = _DupFd

    def __init__(self, process_obj):
        sys.stdout.flush()
        sys.stderr.flush()
        self.returncode = None
        self._fds = []
        self._launch(process_obj)

    if sys.version_info < (3, 4):
        @classmethod
        def duplicate_for_child(cls, fd):
            popen = get_spawning_popen()
            popen._fds.append(fd)
            return reduction._mk_inheritable(fd)

    else:
        def duplicate_for_child(self, fd):
            self._fds.append(fd)
            return reduction._mk_inheritable(fd)

    def poll(self, flag=os.WNOHANG):
        if self.returncode is None:
            while True:
                try:
                    pid, sts = os.waitpid(self.pid, flag)
                except OSError as e:
                    # Child process not yet created. See #1731717
                    # e.errno == errno.ECHILD == 10
                    return None
                else:
                    break
            if pid == self.pid:
                if os.WIFSIGNALED(sts):
                    self.returncode = -os.WTERMSIG(sts)
                else:
                    assert os.WIFEXITED(sts)
                    self.returncode = os.WEXITSTATUS(sts)
        return self.returncode

    def wait(self, timeout=None):
        if sys.version_info < (3, 3):
            import time
            if timeout is None:
                return self.poll(0)
            deadline = time.time() + timeout
            delay = 0.0005
            while 1:
                res = self.poll()
                if res is not None:
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                delay = min(delay * 2, remaining, 0.05)
                time.sleep(delay)
            return res

        if self.returncode is None:
            if timeout is not None:
                from multiprocessing.connection import wait
                if not wait([self.sentinel], timeout):
                    return None
            # This shouldn't block if wait() returned successfully.
            return self.poll(os.WNOHANG if timeout == 0.0 else 0)
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                if self.wait(timeout=0.1) is None:
                    raise

    def _launch(self, process_obj):

        tracker_fd = semaphore_tracker._semaphore_tracker._fd

        fp = BytesIO()
        set_spawning_popen(self)
        if sys.version_info[:2] > (3, 3):
            context.set_spawning_popen(self)
        try:
            prep_data = spawn.get_preparation_data(process_obj._name)
            reduction.dump(prep_data, fp)
            reduction.dump(process_obj, fp)

        finally:
            set_spawning_popen(None)
            if sys.version_info[:2] > (3, 3):
                context.set_spawning_popen(None)

        try:
            parent_r, child_w = os.pipe()
            child_r, parent_w = os.pipe()
            self._chan = CommunicationChannels(parent_w, parent_r)
            # for fd in self._fds:
            #     _mk_inheritable(fd)

            cmd_python = [sys.executable, '-m', 'loky.backend.popen_exec']
            cmd_python += ['--pipe',
                           str(reduction._mk_inheritable(child_w)),
                           str(reduction._mk_inheritable(child_r))]
            if tracker_fd is not None:
                cmd_python += ['--semaphore',
                               str(reduction._mk_inheritable(tracker_fd))]
            util.debug("launch python with cmd:\n%s" % cmd_python)
            self._fds.extend([child_r, child_w, tracker_fd])
            # print('Fd from main:')
            # os.system('ls -l /proc/{}/fd'.format(os.getpid()))
            # print('SEM from main:')
            # os.system('grep shm /proc/{}/maps'.format(os.getpid()))
            from .fork_exec import fork_exec
            self._proc = fork_exec(cmd_python, self._fds)
            self.sentinel = parent_r
            method = 'getbuffer'
            if not hasattr(fp, method):
                method = 'getvalue'
            self._chan._dump(getattr(fp, method)(), self._chan.conn_out,
                             self._chan.pipe_fdw)
            self.pid = self._proc.pid
        finally:
            if parent_r is not None:
                util.Finalize(self, self._chan.close, ())
            for fd in (child_r, child_w):
                if fd is not None:
                    os.close(fd)

    @staticmethod
    def thread_is_spawning():
        return True


class CommunicationChannels(object):
    '''Bi directional communication channel
    '''
    def __init__(self, conn_out, conn_in, strat='buff'):
        self.strat = strat
        self.pipe_fdw = os.fdopen(conn_out, 'wb')
        self.pipe_fdr = os.fdopen(conn_in, 'rb')
        self.conn_out = reduction.ExecPickler(self.pipe_fdw)
        self.conn_in = pickle.Unpickler(self.pipe_fdr)

    def close(self):
        self.pipe_fdw = self._close(self.pipe_fdw)
        self.pipe_fdr = self._close(self.pipe_fdr)

    def _close(self, fh):
        if fh is not None:
            fh.close()
            return None

    def dump(self, obj, conn=None, pipe=None):
        conn_out = conn or self.conn_out
        pipe_fdw = pipe or self.pipe_fdw
        if self.strat == 'string':
            text = pickle.dumps(obj)
            self._dump(text, conn_out, pipe_fdw)
        elif self.strat == 'buff':
            buf = BytesIO()
            reduction.ExecPickler(buf).dump(obj)
            method = 'getbuffer'
            if not hasattr(buf, method):
                method = 'getvalue'
            self._dump(getattr(buf, method)(),
                       conn_out, pipe_fdw)
        elif self.strat == 'pipe':
            conn_out.dump(obj)
            pipe_fdw.flush()

        else:
            raise NotImplementedError('Wrong dump strategy')

    def _dump(self, obj, conn, pipe):
        pipe.write(obj)
        pipe.flush()

    def load(self):
        return self.conn_in.load()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Command line parser')
    parser.add_argument('--pipe', type=int, nargs=2, default=None,
                        help='File handle numbers for the pipe')
    parser.add_argument('--semaphore', type=int, default=None,
                        help='File handle name for the semaphore tracker')
    parser.add_argument('--strat', type=str, default='buff',
                        help='Strategy for communication dump')

    args = parser.parse_args()

    info = dict()
    w, r = args.pipe
    if sys.platform == 'win32':
        import msvcrt
        w = msvcrt.open_osfhandle(w, os.O_WRONLY)
        r = msvcrt.open_osfhandle(r, os.O_RDONLY)
    else:
        semaphore_tracker._semaphore_tracker._fd = args.semaphore
    chan = CommunicationChannels(w, r, strat=args.strat)
    info['backend'] = 'pipe'

    try:
        from multiprocessing import context
        from .process import ExecContext
        context._concrete_contexts['exec'] = ExecContext()
    except ImportError:
        pass

    exitcode = 1
    try:
        prep_data = chan.load()
        spawn.prepare(prep_data)
        process_obj = chan.load()
        # print('Fd from child:')
        # os.system('ls -l /proc/{}/fd'.format(os.getpid()))
        # print('Sem from child:')
        # os.system('grep shm /proc/{}/maps'.format(os.getpid()))
        # print('\n')
        exitcode = process_obj._bootstrap()
    except Exception as e:
        print('\n\n'+'-'*80)
        print('Process failed with traceback: ')
        print('-'*80)
        import traceback
        print(traceback.format_exc())
        print('\n'+'-'*80)
    finally:
        chan.close()
        util.debug('proper close')

        sys.exit(exitcode)
