set DEFAULT_PYTEST_ARGS=-vlx --skip-high-memory

call activate %VIRTUALENV%

pytest %DEFAULT_PYTEST_ARGS%
