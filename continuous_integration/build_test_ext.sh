
cd tests/_openmp
COVERAGE_PROCESS_START=''
test $PYENV != py36 && python setup.py build_ext -i
cd ..