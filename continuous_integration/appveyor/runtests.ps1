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
        python ./continuous_integration/appveyor/tox -e py$ver -- -vlx --timeout=30
        If( $LASTEXITCODE -ne 0){
            Exit 1
        }
    }
    Exit 0
}

TestPythonVersions
