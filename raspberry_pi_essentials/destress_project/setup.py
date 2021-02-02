"""
This script should be included in the root directory.
For this component. the root directory can be considered as follows,
    It is the folder which contains the venv, speech_analysis_raspi and setup.py.

venv - the python virtual environment
speech_analysis_raspi - the one single folder which wraps the full component code
.
├── speech_analysis_raspi
│   ├── mains-predictions
│   │   ├──
│   │   ├──
│   │   └── __init__.py
│   ├── support
│   │   ├──
│   │   ├──
│   │   └── __init__.py
│   └──
│       ├──
│       └── __init__.py
├── setup.py
└── venv

Make sure this setup.py is in the root directory (for this it is "destress_project"),
    the execute "pip install -e ." - . refers to the current directory
"""
from setuptools import setup, find_packages

setup(name='speech_analysis_raspi', version='1.0', packages=find_packages())
