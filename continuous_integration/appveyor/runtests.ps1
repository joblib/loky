# This script is meant to be called by the "test" step defined in
# appveyor.yml. See http://appveyor.com/docs for more details.
# Authors: Thomas Moreau
# License: 3-clause BSD

$VERSION=(36, 34, 27)

function TestPythonVersions () {
    Write-Host $PYTHON
    ForEach($ver in $VERSION){
        $env:TOXPYTHON = "C:\Python$ver$env:PYTHON_ARCH_SUFFIX\python.exe"
        Write-Host $env:TOXPYTHON
        $PYTEST_ARGS = "-vlx --timeout=50"
        If( $env:PYTHON_ARCH -eq 32){
            $PYTEST_ARGS = "$PYTEST_ARGS --no-memory"
        }
        python ./continuous_integration/appveyor/tox -e py$ver -- $PYTEST_ARGS
        If( $LASTEXITCODE -ne 0){
            Exit 1
        }
    }
    Exit 0
}

TestPythonVersions
