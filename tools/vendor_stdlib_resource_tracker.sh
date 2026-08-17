#!/bin/sh

curl -L \
     https://raw.githubusercontent.com/python/cpython/refs/tags/v3.14.7/Lib/multiprocessing/resource_tracker.py \
     -o loky/backend/stdlib_py314_resource_tracker.py

sed -i 's@from \. import@from multiprocessing import@' loky/backend/stdlib_py*_resource_tracker.py
