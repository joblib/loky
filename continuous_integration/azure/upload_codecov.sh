#!/bin/bash

set -e

pip install codecov

coverage combine --append
codecov || echo "codecov upload failed"
