#!/bin/bash
source ../BaseStack/bin/setup_run.sh
PYTHONPATH=`pwd`/src:${PYTHONPATH}
export PYTHONPATH
source sub/bin/activate
