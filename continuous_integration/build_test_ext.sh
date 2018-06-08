
cd tests/_openmp_test_helper
COVERAGE_PROCESS_START=''
python setup.py build_ext -i || echo 'No openmp'
cd ../..
