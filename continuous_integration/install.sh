#!/bin/bash

# License: 3-clause BSD

set -xe

# Create new conda env
conda config --set solver libmamba

if [[ "$FREE_THREADING" == "true" ]]; then
    PYTHON_PACKAGE="python-freethreading"
else
    PYTHON_PACKAGE="python"
fi

# If the conda channel is not explicitly set, use conda-forge:
if [[ -z "$CONDA_CHANNEL" ]]; then
    CONDA_CHANNEL="conda-forge"
fi

to_install="$PYTHON_PACKAGE=$PYTHON_VERSION pip"
# to_install="$PYTHON_PACKAGE=$PYTHON_VERSION pip numpy tblib $EXTRA_PACKAGES"
conda create -n testenv --yes -c $CONDA_CHANNEL $to_install
conda activate testenv

if [[ -z "$JOBLIB_TESTS" ]]; then
    # Install pytest timeout to fasten failure in deadlocking tests
    PIP_INSTALL_PACKAGES="pytest pytest-timeout coverage pytest-cov"
fi

if [[ -z "$NO_PSUTIL" ]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES psutil"
fi

pip install numpy $PIP_INSTALL_PACKAGES

pip install -v .
