#!/bin/sh
# Change to this directory
cd `echo $0 | sed -e 's/[^/]*$//'`
cd ..
pytest
cd dev

echo '=== autopep8'
autopep8 -i --aggressive ../src/multiai_tts/*.py

echo '=== flake8'
flake8 --ignore=E501,W504 ../src/multiai_tts/*.py
