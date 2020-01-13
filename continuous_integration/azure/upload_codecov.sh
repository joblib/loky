#!/bin/bash

set -e

# Newer versions of coverage trigger a DLL error involving sqlite3 on Windows
pip install coverage
pip install codecov

COVERAGE_STORAGE=json coverage combine --append
COVERAGE_STORAGE=json codecov || echo "codecov upload failed"
