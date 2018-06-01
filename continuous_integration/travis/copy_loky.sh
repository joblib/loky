#!/bin/sh
# Script to do a local install of loky
set +x
export LC_ALL=C
INSTALL_FOLDER=tmp/loky_install
rm -rf loky $INSTALL_FOLDER
if [ -z "$1" ]
then
        LOKY=loky
else
        LOKY=$1
fi

pip install $LOKY --target $INSTALL_FOLDER
cp -r $INSTALL_FOLDER/loky .
rm -rf $INSTALL_FOLDER

# Needed to rewrite the doctests
# Note: BSD sed -i needs an argument unders OSX
# so first renaming to .bak and then deleting backup files
find loky -name "*.py" | xargs sed -i.bak "s/from loky/from joblib.externals.loky/"
find loky -name "*.bak" | xargs rm
