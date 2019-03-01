set DEFAULT_PYTEST_ARGS=-vlx --skip-high-memory

call activate %VIRTUALENV%

python continuous_integration/install_coverage_subprocess_pth.py

pytest --junitxml=%JUNITXML% %DEFAULT_PYTEST_ARGS%
