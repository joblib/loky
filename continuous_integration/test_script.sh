#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

set -e

ver=$(python -V 2>&1 | sed -e 's/Python \([23]\.[0-9]\).*/\1/' -e 's/\.//')
AUXFILE=.aux$ver
DEADLOCK=.exit_on_lock

python --version
py.test -vsx 2>$AUXFILE
[ $? -ne 0 ] &&cat $AUXFILE
[ -e "$DEADLOCK" ] && cat $DEADLOCK
rm $AUXFILE
