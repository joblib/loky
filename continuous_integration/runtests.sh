#!/bin/bash

# License: 3-clause BSD

set -xe

conda activate testenv

which python
python -V
python -c "import struct; print('platform: %d' % (8 * struct.calcsize('P')))"
python -c "import loky; print('loky.cpu_count():', loky.cpu_count())"
python -c "import os; print('os.cpu_count():', os.cpu_count())"


if [[ "$JOBLIB_TESTS" == "true" ]]; then
    # Install joblib from pip, patch it to use this version of loky
    # and run the joblib tests with pytest.
    LOKY_PATH=$(pwd)

    git clone https://github.com/joblib/joblib.git src_joblib
    cd src_joblib
    pip install "pytest<7.0"  # Need to update remove occurrences of pytest.warns(None)
    pip install threadpoolctl  # required by some joblib tests

    pip install -e .
    export JOBLIB=`python -c "import joblib; print(joblib.__path__[0])"`
    cp "$LOKY_PATH"/continuous_integration/copy_loky.sh $JOBLIB/externals
    (cd $JOBLIB/externals && bash copy_loky.sh "$LOKY_PATH")
    pytest -vl --ignore $JOBLIB/externals --pyargs joblib
else
    # Make sure that we have the python docker image cached locally to avoid
    # a timeout in a test that needs it.
    if [[ "$(which docker)" != "" ]] && [[ "$(uname)" = "Linux" ]]; then
        docker pull python:3.10
    fi

    # Enable coverage reporting from subprocesses.
    python continuous_integration/install_coverage_subprocess_pth.py

    PYTEST_ARGS="-vl --timeout=120 --maxfail=5 --cov=loky --cov-report xml"

    if [[ "$RUN_MEMORY" != "true" ]]; then
        PYTEST_ARGS="$PYTEST_ARGS --skip-high-memory"
    fi

    LOKY_MAX_DEPTH=3
    OMP_NUM_THREADS=4

    # Run the tests and collect trace coverage data both in the subprocesses
    # and its subprocesses.
    pytest $PYTEST_ARGS .
fi
