import loky


def test_version():
    assert hasattr(loky, '__version__'), (
        "There are no __version__ argument on the loky module")