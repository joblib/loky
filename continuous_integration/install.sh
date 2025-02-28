#!/bin/bash

# License: 3-clause BSD

set -xe

# Create new conda env
conda config --set solver libmamba
to_install="python=$PYTHON_VERSION pip numpy tblib viztracer"
conda create -n testenv --yes -c conda-forge $to_install
conda activate testenv

if [[ -z "$JOBLIB_TESTS" ]]; then
    # Install pytest timeout to fasten failure in deadlocking tests
    PIP_INSTALL_PACKAGES="pytest pytest-timeout coverage pytest-cov"
fi

if [[ -z "$NO_PSUTIL" ]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES psutil"
fi

pip install $PIP_INSTALL_PACKAGES

pip install -v .
