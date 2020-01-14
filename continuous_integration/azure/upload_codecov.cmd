call activate %VIRTUALENV%
pip install codecov
coverage combine --append
codecov || echo "codecov upload failed"
