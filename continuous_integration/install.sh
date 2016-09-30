#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

set -e

PIP="pip"
test $PYTHON == "python3" && PIP="pip3"

ver=$($PYTHON -V 2>&1 | sed -e 's/Python \([23]\.[0-9]\).*/\1/' -e 's/\.//')
echo "Testing for python $ver"

# Make sure to install a recent pip version to be able to install numpy
# with built-in openblas under linux
$PIP install --upgrade pip
$PIP uninstall -y numpy || echo "numpy not previously installed"
$PIP install psutil pytest

# numpy is not available as wheel in Python 3.3
[ $ver -lt 33 ] && $PIP install numpy
[ $ver -gt 33 ] && $PIP install numpy

# Backport modules for Python 2.7
[ $ver -lt 33 ] && $PIP install faulthandler
[ $ver -lt 33 ] && $PIP install futures

$PYTHON setup.py develop
