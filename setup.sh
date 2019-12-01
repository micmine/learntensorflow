#!/bin/bash

rm -r ./venv            
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow

#deactivate