set DEFAULT_PYTEST_ARGS="-vlx --timeout=50 --skip-high-memory"

call activate %VIRTUALENV%

pytest %DEFAULT_PYTEST_ARGS%
