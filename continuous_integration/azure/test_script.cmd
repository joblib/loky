set DEFAULT_PYTEST_ARGS=-vlrx --timeout=60 --cov loky --skip-high-memory

call activate %VIRTUALENV%

python continuous_integration/install_coverage_subprocess_pth.py

pytest --junitxml=%JUNITXML% %DEFAULT_PYTEST_ARGS%
