#!/bin/bash

set -e

# Newer versions of coverage trigger a DLL error involving sqlite3 on Windows
pip install 'coverage < 5'
pip install codecov

coverage combine --append
codecov || echo "codecov upload failed"
