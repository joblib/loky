#!/bin/bash

# License: 3-clause BSD

set -xe

which python
python -V
python -c "import struct; print('platform: %d' % (8 * struct.calcsize('P')))"
python -c "import loky; print('loky.cpu_count():', loky.cpu_count())"
python -c "import os; print('os.cpu_count():', os.cpu_count())"


run_pytest_with_coverage() {
    local pytest_status=0 coverage_status=0
    pytest "$@" || pytest_status=$?

    # Trackers can save coverage during interpreter shutdown, after pytest's
    # report. Regenerating XML also combines these late parallel data files.
    python -m coverage xml || coverage_status=$?
    if [[ "$pytest_status" -ne 0 ]]; then
        return "$pytest_status"
    fi
    return "$coverage_status"
}


if [[ "$JOBLIB_TESTS" == "true" ]]; then
    # Install joblib from pip, patch it to use this version of loky
    # and run the joblib tests with pytest.
    LOKY_PATH=$(pwd)

    git clone https://github.com/joblib/joblib.git src_joblib
    cd src_joblib
    pip install --no-deps -e .

    export JOBLIB=`python -c "import joblib; print(joblib.__path__[0])"`
    # Need to copy root conftest.py to inside joblib package since we are using
    # pytest --pyargs joblib a few lines below
    cp conftest.py $JOBLIB/conftest.py
    # Update vendored loky code inside joblib
    cp "$LOKY_PATH"/continuous_integration/copy_loky.sh $JOBLIB/externals
    (cd $JOBLIB/externals && bash copy_loky.sh "$LOKY_PATH")
    pytest -vl --ignore $JOBLIB/externals --pyargs joblib
elif [[ "$PYINSTALLER_TESTS" == "true" ]]; then
    PYTEST_ARGS="-vl --timeout=300 --maxfail=1 --cov=loky --cov-report xml"
    run_pytest_with_coverage $PYTEST_ARGS tests/test_loky_module.py -k freeze
else
    # Make sure that we have the python docker image cached locally to avoid
    # a timeout in a test that needs it.
    if [[ "$(which docker)" != "" ]] && [[ "$(uname)" = "Linux" ]]; then
        docker pull python:3.10
    fi

    PYTEST_ARGS="-vl --timeout=120 --maxfail=5 --cov=loky --cov-report xml"

    if [[ "$RUN_MEMORY" != "true" ]]; then
        PYTEST_ARGS="$PYTEST_ARGS --skip-high-memory"
    fi

    LOKY_MAX_DEPTH=3
    OMP_NUM_THREADS=4

    # Run the tests and collect trace coverage data both in the subprocesses
    # and its subprocesses.
    run_pytest_with_coverage $PYTEST_ARGS .
fi
