#!/usr/bin/env sh
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

set -e
$PYTHON --version
echo $TOXENV

# Run the tests and collect trace coverage data both in the subprocesses
# and its subprocesses.
COVERAGE_PROCESS_START="$TRAVIS_BUILD_DIR/.coveragerc" tox -- -vl \
        --timeout=30 --maxfail=5
