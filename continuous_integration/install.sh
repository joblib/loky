#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

set -e

ver=$(python -V 2>&1 | sed -e 's/Python \([23]\.[0-9]\).*/\1/' -e 's/\.//')
echo "Testing for python $ver"
pip install psutil
[ $ver -lt 33 ] && pip install faulthandler
[ $ver -lt 33 ] && pip install futures

python setup.py develop 
