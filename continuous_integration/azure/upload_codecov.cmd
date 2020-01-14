call activate %VIRTUALENV%
where pip

pip install codecov
where codecov
where coverage

coverage combine --append
codecov
