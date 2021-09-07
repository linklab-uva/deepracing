#!/bin/bash
scriptdir=$(dirname $(readlink -f "$0"))
echo $scriptdir
extrapythonpaths=$scriptdir/deepracing_py:$scriptdir/DCNN-Pytorch
echo $extrapythonpaths

# if [[ -z "${PYTHONPATH}" ]]; then
#   export PYTHONPATH=${extrapythonpaths}
# else
#   export PYTHONPATH=${PYTHONPATH}:${extrapythonpaths}
# fi
