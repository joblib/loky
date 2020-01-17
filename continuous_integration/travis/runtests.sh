#!/usr/bin/env sh
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

export
set -e
which python
python --version
echo ${TOX_ENV}

# Make sure that we have the python docker image cached locally to avoid
# a timeout in a test that needs it.

if [ "$(which docker)" != "" ] && [ "$(uname)" = "Linux" ]; then
    docker pull python:3.6
fi

if [ "$JOBLIB_TESTS" = "true" ]; then
    # Install joblib from pip, patch it to use this version of loky
    # and run the joblib tests with pytest.
    git clone https://github.com/joblib/joblib.git src_joblib
    cd src_joblib
    pip install pytest
    pip install threadpoolctl  # required by some joblib tests

    pip install -e .
    export JOBLIB=`python -c "import joblib; print(joblib.__path__[0])"`
    cp $BUILD_SOURCESDIRECTORY/continuous_integration/travis/copy_loky.sh $JOBLIB/externals
    (cd $JOBLIB/externals && bash copy_loky.sh $BUILD_SOURCESDIRECTORY)
    pytest -vl --ignore $JOBLIB/externals --pyargs joblib
else
    # Run the tests and collect trace coverage data both in the subprocesses
    # and its subprocesses.
    PYTEST_ARGS="-vl --timeout=60 --maxfail=5"
    if [ "$RUN_MEMORY" != "true" ]; then
        PYTEST_ARGS="$PYTEST_ARGS --skip-high-memory"
    fi
    tox -v -e ${TOX_ENV}
fi
