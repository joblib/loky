deactivate
conda remove --all -q -y -n $VIRTUALENV
conda create -n $VIRTUALENV -q -y python=$VERSION_PYTHON
conda activate $VIRTUALENV
which python
python -c "import sys; print(sys.platform)"
