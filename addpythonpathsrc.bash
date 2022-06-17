#!/bin/bash
pushd . > /dev/null
SCRIPT_PATH="${BASH_SOURCE[0]}";
if ([ -h "${SCRIPT_PATH}" ]) then
  while([ -h "${SCRIPT_PATH}" ]) do cd `dirname "$SCRIPT_PATH"`; SCRIPT_PATH=`readlink "${SCRIPT_PATH}"`; done
fi
cd `dirname ${SCRIPT_PATH}` > /dev/null
SCRIPT_PATH=`pwd`;
popd  > /dev/null

extrapythonpaths=$SCRIPT_PATH/deepracing_py:$SCRIPT_PATH/DCNN-Pytorch
echo "Adding $extrapythonpaths to PYTHONPATH"

if [[ -z "${PYTHONPATH}" ]]; then
  export PYTHONPATH=${extrapythonpaths}
else
  export PYTHONPATH=${extrapythonpaths}:${PYTHONPATH}
fi
