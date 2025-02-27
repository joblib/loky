#!/bin/bash

# License: 3-clause BSD

set -xe

conda activate testenv

which python
python -V
python -c "import struct; print('platform: %d' % (8 * struct.calcsize('P')))"
python -c "import loky; print('loky.cpu_count():', loky.cpu_count())"
python -c "import os; print('os.cpu_count():', os.cpu_count())"


if [ "$JOBLIB_TESTS" = "true" ]; then
    # Install joblib from pip, patch it to use this version of loky
    # and run the joblib tests with pytest.
    python -m venv venv/
    source ./venv/bin/activate
    which python
    git clone https://github.com/joblib/joblib.git src_joblib
    cd src_joblib
    pip install "pytest<7.0"  # Need to update remove occurrences of pytest.warns(None)
    pip install threadpoolctl  # required by some joblib tests

    pip install -e .
    export JOBLIB=`python -c "import joblib; print(joblib.__path__[0])"`
    cp "$BUILD_SOURCESDIRECTORY"/continuous_integration/copy_loky.sh $JOBLIB/externals
    (cd $JOBLIB/externals && bash copy_loky.sh "$BUILD_SOURCESDIRECTORY")
    pytest -vl --ignore $JOBLIB/externals --pyargs joblib
else
    # Make sure that we have the python docker image cached locally to avoid
    # a timeout in a test that needs it.
    if [ "$(which docker)" != "" ] && [ "$(uname)" = "Linux" ]; then
        docker pull python:3.10
    fi

    # Run the tests and collect trace coverage data both in the subprocesses
    # and its subprocesses.
    if [ "$RUN_MEMORY" != "true" ]; then
        PYTEST_ARGS="$PYTEST_ARGS --skip-high-memory"
    fi
    tox -v -e "${TOX_ENV}"  -- ${PYTEST_ARGS} --junitxml="${JUNITXML}"
fi





if [[ "$SKLEARN_TESTS" != "true" ]]; then
    pytest joblib -vl --timeout=120 --cov=joblib --cov-report xml

    # doctests are not compatile with default_backend=threading
    if [[ "$JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND" != "threading" ]]; then
        make test-doc
    fi
else
    # Install the nightly build of scikit-learn and test against the installed
    # development version of joblib.
    conda install -y -c conda-forge cython numpy scipy
    pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn
    python -c "import sklearn; print('Testing scikit-learn', sklearn.__version__)"

    # Move to a dedicated folder to avoid being polluted by joblib specific conftest.py
    # and disable the doctest plugin to avoid issues with doctests in scikit-learn
    # docstrings that require setting print_changed_only=True temporarily.
    NEW_TEST_DIR=$(mktemp -d)
    cd $NEW_TEST_DIR

    pytest -vl --maxfail=5 -p no:doctest \
        --pyargs sklearn
fi
