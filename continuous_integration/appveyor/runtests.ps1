# This script is meant to be called by the "test" step defined in
# appveyor.yml. See http://appveyor.com/docs for more details.
# Authors: Thomas Moreau
# License: 3-clause BSD

$VERSION=(36, 27)
$TOX_CMD = "python ./continuous_integration/appveyor/tox"

function TestPythonVersions () {
    Write-Host $PYTHON
    ForEach($ver in $VERSION){
        $env:TOXPYTHON = "C:\Python$ver$env:PYTHON_ARCH_SUFFIX\python.exe"
        Write-Host $env:TOXPYTHON
        # Skip memory test as the appveyor environment is too small for those.
        $PYTEST_ARGS = "-vl --timeout=60 --maxfail=5 --skip-high-memory"

        # Launch the tox command for the correct python version. We use `iex`
        # to correctly pass PYTEST_ARGS, which are parsed as files otherwise.
        iex "$TOX_CMD -e py$ver -- $PYTEST_ARGS"
        If( $LASTEXITCODE -ne 0){
            Exit 1
        }
    }
    Exit 0
}

TestPythonVersions
