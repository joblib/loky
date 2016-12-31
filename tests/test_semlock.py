# -*- coding: utf-8 -*-
# @Author: Thomas Moreau
# @Date:   2016-12-31 12:05:45
# @Last Modified by:   Thomas Moreau
# @Last Modified time: 2016-12-31 12:26:39
import sys
import pytest

if sys.version_info < (3, 3):
    FileExistsError = OSError
    FileNotFoundError = OSError


@pytest.mark.skipif(sys.platform == "win32", reason="UNIX test")
def test_semlock_failure():
    from loky.backend.semlock import SemLock, sem_unlink
    name = "test1"
    sl = SemLock(0, 1, 1, name=name)

    with pytest.raises(FileExistsError):
        SemLock(0, 1, 1, name=name)
    sem_unlink(sl.name)

    with pytest.raises(FileNotFoundError):
        SemLock._rebuild(None, 0, 0, name.encode('ascii'))
