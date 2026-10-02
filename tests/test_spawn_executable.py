import os
import sys

import pytest
from multiprocessing import util

from loky.backend.resource_tracker import spawnv_passfds
from loky.backend.spawn import get_executable


def test_get_executable_returns_bytes_on_posix():
    exe = get_executable()
    if sys.platform == "win32":
        assert isinstance(exe, str)
        assert exe == sys.executable
    else:
        assert isinstance(exe, bytes)
        assert exe == os.fsencode(sys.executable)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX encoding path")
def test_spawnv_passfds_accepts_bytes_path(monkeypatch):
    seen = {}

    def fake_spawnv(path, args, passfds):
        seen["path"] = path
        return 17

    monkeypatch.setattr(util, "spawnv_passfds", fake_spawnv)

    pid, handle = spawnv_passfds(b"/usr/bin/python", [b"/usr/bin/python"], [])
    assert pid == 17
    assert handle is None
    assert seen["path"] == b"/usr/bin/python"
