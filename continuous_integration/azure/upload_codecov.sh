#!/bin/bash

set -e

conda activate $VIRTUALENV
which pip

pip install coverage
pip install codecov
which codecov
which coverage

coverage combine --append
codecov || echo "codecov upload failed"
