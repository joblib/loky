#!/bin/sh

# Download the CPython stdlib reference file. The CPython version can be
# updated from time to time
curl -L \
     https://raw.githubusercontent.com/python/cpython/refs/tags/v3.14.7/Lib/multiprocessing/resource_tracker.py \
     -o loky/backend/stdlib_py314_resource_tracker.py.ref

# Overwrite the stdlib_py314_resource_tracker.py by applying our custom patch
# to the stdlib reference file
patch \
     loky/backend/stdlib_py314_resource_tracker.py.ref \
     -i tools/stdlib_py314_resource_tracker.patch \
     -o loky/backend/stdlib_py314_resource_tracker.py

# Delete the stdlib reference file
rm -f loky/backend/stdlib_py314_resource_tracker.py.ref
