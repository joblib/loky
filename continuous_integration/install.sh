#!/bin/bash

# License: 3-clause BSD

set -xe

# Create new conda env
conda config --set solver libmamba
to_install="python=$PYTHON_VERSION pip pytest numpy tblib viztracer"
conda create -n testenv --yes -c conda-forge $to_install
conda activate testenv

# Install pytest timeout to fasten failure in deadlocking tests
PIP_INSTALL_PACKAGES="pytest-timeout coverage pytest-cov"

if [[ -z "$NO_PSUTIL"]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES psutil"
fi

pip install $PIP_INSTALL_PACKAGES

pip install -v .
