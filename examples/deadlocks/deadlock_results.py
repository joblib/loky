"""Deadlock with unpickling error for the result
=================================================

This example highlights the fact that the ProcessPoolExecutor implementation
from concurrent.futures is not robust to pickling error (at least in versions
3.6 and lower).
"""
import argparse
from pickle import UnpicklingError


def return_instance(cls):
    """Function that returns a instance of cls"""
    return cls()


def raise_error(Err):
    """Function that raises an Exception in process"""
    raise Err()


class ObjectWithPickleError():
    """Triggers a RuntimeError when sending job to the workers"""

    def __reduce__(self):
        return raise_error, (UnpicklingError, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-stdlib', action="store_true",
                        help='Use concurrent.futures.ProcessPoolExecutor'
                             ' instead of loky')
    args = parser.parse_args()
    if args.use_stdlib:
        from concurrent.futures import ProcessPoolExecutor
    else:
        from loky import ProcessPoolExecutor

    with ProcessPoolExecutor() as e:
        f = e.submit(return_instance, ObjectWithPickleError)
        f.result()
