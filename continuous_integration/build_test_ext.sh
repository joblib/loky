
cd tests/_openmp
COVERAGE_PROCESS_START=''
python setup.py build_ext -i || echo 'No openmp'
cd ../..