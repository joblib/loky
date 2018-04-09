"""
Deadlock on failed pickle
=========================

This example highlights the fact that the ProcessPoolExecutor implementation
from concurrent.futures is not robust to pickling error (at least in versions
3.6 and lower).
"""
import argparse


class ObjectWithPickleError():
    """Triggers a RuntimeError when sending job to the workers"""

    def __reduce__(self):
        raise RuntimeError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-stdlib', action="store_true",
                        help='Use concurrent.futures.ProcessPoolExecutor'
                             ' instead of loy')
    args = parser.parse_args()
    if args.use_stdlib:
        from concurrent.futures import ProcessPoolExecutor
    else:
        from loky import ProcessPoolExecutor

    with ProcessPoolExecutor() as e:
        f = e.submit(id, ObjectWithPickleError()).result()
