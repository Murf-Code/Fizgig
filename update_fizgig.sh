#!/bin/bash
# Updates Fizgig (NVIDIA): git pull, then the dependencies. All the logic lives in update_fizgig.py.
# AMD ROCm on Linux: run ./install_fizgig_rocm.sh again instead - this one installs CUDA packages.
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    git pull
    echo
    echo "WARNING: venv not found - run python3 install_fizgig.py to set it up."
    exit 1
fi
exec venv/bin/python update_fizgig.py "$@"
