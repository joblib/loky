#!/bin/sh

# TODO Remove the Python 3.10 file when we get rid of Python 3.10 support.
# Python 3.10 is end-of-life is 31 October 2026.
curl -L \
     https://raw.githubusercontent.com/python/cpython/refs/tags/v3.10.20/Lib/multiprocessing/resource_tracker.py \
     -o loky/backend/stdlib_py310_resource_tracker.py
curl -L \
     https://raw.githubusercontent.com/python/cpython/refs/tags/v3.14.7/Lib/multiprocessing/resource_tracker.py \
     -o loky/backend/stdlib_py314_resource_tracker.py

sed -i 's@from \. import@from multiprocessing import@' loky/backend/stdlib_py*_resource_tracker.py
