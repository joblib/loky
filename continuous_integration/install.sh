#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

set -e

PIP="pip"
test $PYTHON == "python3" && PIP="pip3"

ver=$($PYTHON -V 2>&1 | sed -e 's/Python \([23]\.[0-9]\).*/\1/' -e 's/\.//')
echo "Testing for python $ver"
$PIP install psutil pytest numpy
[ $ver -lt 33 ] && $PIP install faulthandler
[ $ver -lt 33 ] && $PIP install futures

$PYTHON setup.py develop
