set DEFAULT_PYTEST_ARGS=-vlx --skip-high-memory

call activate %VIRTUALENV%

pytest --junitxml=%JUNITXML% %DEFAULT_PYTEST_ARGS%
coverage combine --append
