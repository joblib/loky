
cd tests/_openmp
COVERAGE_PROCESS_START=''
if [[ ${TRAVIS_OS_NAME} = osx ]]; then
    # Since default gcc on osx is just a front-end for LLVM
    CC=gcc-4.8 python setup.py build_ext -i
elif [[ $PYENV != py36 ]]; then
	python setup.py build_ext -i
fi
cd ../..