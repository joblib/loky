#!/bin/bash

set -e

conda activate $VIRTUALENV

pip install coverage
pip install codecov

coverage combine --append
codecov || echo "codecov upload failed"
