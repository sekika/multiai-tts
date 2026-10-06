#!/bin/sh
# Change to this directory
cd `echo $0 | sed -e 's/[^/]*$//'`

# test
./test.sh

# Make package and upload with the same Python environment.  Recent
# hatchling emits Core Metadata 2.5, which older standalone ``twine``
# executables cannot parse.
python3 -m pip install --upgrade build twine
echo "Making packages."
cd ..
python3 -m build

# Token required. Check ~/.pypirc
python3 -m twine upload --skip-existing dist/*

# Uninstall multiai
python3 -m pip uninstall multiai-tts
echo "Upload completed. Installed version uninstalled. Wait for a while and run"
echo "python3 -m pip install multiai-tts"
