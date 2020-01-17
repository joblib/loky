set -o errexit

deactivate
conda remove --all -q -y -n $VIRTUALENV
conda create -n $VIRTUALENV -q -y python=$VERSION_PYTHON
conda activate $VIRTUALENV
which python
python --version
python -c "import sys; print(sys.platform)"
python -m pip install -U pip
pip --version

if [[ $PACKAGER == "conda" ]]; then
    conda install numpy=1.15 psutil pytest cython
fi

if [[ $PACKAGER == "pip" ]]; then
    python -m pip install -r dev-requirements.txt
fi

python -m pip install -q pytest-timeout coverage pytest-coverage

python -m pip install -e .

# Build the cython test helper for openmp
bash ./continuous_integration/build_test_ext.sh
