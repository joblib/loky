#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version
AUXFILE=.aux27
for version in {27,33,34}
do
    tox -e py${version} 2>$AUXFILE
    if [[ $? -ne 0 ]]
    then
        cat $AUXFILE
        break
    fi
done
rm $AUXFILE
