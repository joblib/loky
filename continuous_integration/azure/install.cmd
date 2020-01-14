@rem https://github.com/numba/numba/blob/master/buildscripts/incremental/setup_conda_environment.cmd
@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

@rem Deactivate any environment
call deactivate
@rem Clean up any left-over from a previous build and install version of python
conda remove --all -q -y -n %VIRTUALENV%
conda create -n %VIRTUALENV% -q -y python=%VERSION_PYTHON%

call activate %VIRTUALENV%
python -m pip install -U pip
python --version
pip --version

@rem Install dependencies with either conda or pip.
if "%PACKAGER%" == "conda" (%CONDA_INSTALL% numpy=1.15 psutil pytest cython)
if "%PACKAGER%" == "pip" (%PIP_INSTALL% -r dev-requirements.txt)

@rem Install extra dependency
pip install -q pytest-timeout coverage pytest-coverage

@rem Install package
pip install -e .

@rem Build the cython test helper for openmp
bash ./continuous_integration/build_test_ext.sh

if %errorlevel% neq 0 exit /b %errorlevel%
