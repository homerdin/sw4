#!/bin/bash

# module paths

export MODULEPATH=$MODULEPATH:/soft/modulefiles
export MODULEPATH=$MODULEPATH:/soft/restricted/CNDA/modulefiles/

module load cmake
module load oneapi
module load mpi


export LD_LIBRARY_PATH=/home/bhomerding/install/proj-5.2.0/lib:$LD_LIBRARY_PATH
