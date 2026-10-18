import functools
import pickle

import pytest

from loky.backend import reduction


@pytest.fixture(params=["pickle", "cloudpickle"])
def pickler(request):
    previous = reduction.get_loky_pickler_name()
    reduction.set_loky_pickler(request.param)
    try:
        yield
    finally:
        reduction.set_loky_pickler(previous)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_partial_instance_state(pickler, protocol):
    callback = functools.partial(pow, exp=2)
    callback.terminal = True
    callback.direction = -1
    callback.metadata = {"labels": ["simulation"]}
    callback.alias = callback.metadata
    callback.self = callback
    callback.__name__ = "event"

    restored = reduction.loads(reduction.dumps(callback, protocol=protocol))

    assert restored(3) == 9
    assert restored.terminal is True
    assert restored.direction == -1
    assert restored.metadata == callback.metadata
    assert restored.metadata is not callback.metadata
    assert restored.alias is restored.metadata
    assert restored.self is restored
    assert restored.__name__ == "event"


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_partial_argument_state_alias(pickler, protocol):
    callback = functools.partial(list, [1, 2])
    callback.metadata = callback.args[0]

    restored = reduction.loads(reduction.dumps(callback, protocol=protocol))

    assert restored() == [1, 2]
    assert restored.metadata is restored.args[0]
    assert restored.metadata is not callback.metadata


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_partial_recursive_keyword(pickler, protocol):
    callback = functools.partial(dict, owner=None)
    callback.keywords["owner"] = callback

    restored = reduction.loads(reduction.dumps(callback, protocol=protocol))

    assert restored()["owner"] is restored
