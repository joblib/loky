#!/usr/bin/env sh
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

set -e
$PYTHON --version
echo $TOXENV

# Make sure that we have the python docker image cached locally to avoid
# a timeout in a test that needs it.

if [ `which docker` != "" ]; then
    docker pull python:3.6
fi

# Run the tests and collect trace coverage data both in the subprocesses
# and its subprocesses.
COVERAGE_PROCESS_START="$TRAVIS_BUILD_DIR/.coveragerc" tox -- -vl \
        --timeout=30 --maxfail=5


if [ "$JOBLIB_TESTS" == "true" ]; then
    # Install joblib from pip, patch it to use this version of loky
    # and run the joblib tests with pytest.
    pip install joblib
    export JOBLIB=`python -c "import joblib; print(joblib.__path__[0])"`
    cp $TRAVIS_BUILD_DIR/continuous_integration/travis/copy_loky.sh $JOBLIB/externals
    (cd $JOBLIB/externals && bash copy_loky.sh $TRAVIS_BUILD_DIR)
    cp $TRAVIS_BUILD_DIR/continuous_integration/travis/conftest.py $JOBLIB/..
    pytest -vl --ignore $JOBLIB/externals --pyargs joblib
fi