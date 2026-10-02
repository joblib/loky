import pickle
import subprocess
import sys
import types
from pickle import loads

import cloudpickle
import pytest

from loky.backend import reduction


class ImportableMethods:
    def instance_method(self, value):
        return value + 1

    @classmethod
    def class_method(cls, value):
        return value + 1


@pytest.fixture
def set_pickler():
    previous = reduction.get_loky_pickler_name()
    try:
        yield reduction.set_loky_pickler
    finally:
        reduction.set_loky_pickler(previous)


@pytest.mark.parametrize("as_instance", [False, True])
def test_dynamic_class_referencing_its_classmethod(set_pickler, as_instance):
    # Mirrors a Pydantic model, which stores its validators in its own
    # metadata (joblib/joblib#1732).
    # A getattr-based reducer tries to resolve the method before the dynamic
    # class's state (including the method) has been restored.
    class DynamicClass:
        @classmethod
        def validate(cls, value):
            return value + 1

    DynamicClass.callbacks = {"validate": DynamicClass.validate}
    value = DynamicClass() if as_instance else DynamicClass
    code = """
import pickle
import sys

obj = pickle.loads(sys.stdin.buffer.read())
callback = obj.callbacks["validate"]
cls = obj if isinstance(obj, type) else type(obj)
assert callback.__self__ is cls
assert callback(41) == 42
"""
    set_pickler("cloudpickle")
    # cloudpickle.dumps is the control: it checks the assertions in the child.
    for dumps in (cloudpickle.dumps, reduction.dumps):
        # The fresh interpreter avoids cloudpickle's dynamic-class cache,
        # which can hide the reconstruction error in a same-process test.
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=bytes(dumps(value)),
            capture_output=True,
            timeout=10,
        )
        assert proc.returncode == 0, proc.stderr.decode()


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("method", ["instance_method", "class_method"])
def test_stdlib_pickler_round_trips_methods(set_pickler, method, protocol):
    # Without a loky-specific reducer, the standard pickler relies on the
    # methods' own __reduce__.
    set_pickler("pickle")
    original = getattr(ImportableMethods(), method)
    restored = loads(reduction.dumps(original, protocol=protocol))
    assert restored(41) == 42


def test_importable_instance_method_keeps_its_function(set_pickler):
    # cloudpickle pickles the function of an importable class by reference.
    set_pickler("cloudpickle")
    restored = loads(reduction.dumps(ImportableMethods().instance_method))
    assert restored.__func__ is ImportableMethods.instance_method


@pytest.mark.parametrize("pickler", ["pickle", "cloudpickle"])
@pytest.mark.parametrize("registration", ["global", "per_call"])
def test_custom_method_reducer_precedence(
    monkeypatch, set_pickler, pickler, registration
):
    set_pickler(pickler)
    original = ImportableMethods.class_method

    def custom_reducer(method):
        return int, (99,)

    with monkeypatch.context() as scoped_patch:
        kwargs = {}
        if registration == "global":
            # Use the public registration API but restore its global state.
            scoped_patch.setattr(
                reduction, "_dispatch_table", reduction._dispatch_table.copy()
            )
            reduction.register(types.MethodType, custom_reducer)
        else:
            kwargs["reducers"] = {types.MethodType: custom_reducer}
        assert loads(reduction.dumps(original, **kwargs)) == 99

    # A custom dump must not change the base pickler's table or leak into
    # subsequent dumps after the registration has been restored.
    assert loads(reduction.dumps(original))(41) == 42
