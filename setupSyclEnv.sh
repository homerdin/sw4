#!/bin/bash

# module paths

export MODULEPATH=$MODULEPATH:/soft/modulefiles
export MODULEPATH=$MODULEPATH:/soft/restricted/CNDA/modulefiles/

module load spack/linux-rhel7-x86_64
module load cmake
module load oneapi
module load openmpi/2.1.6-intel

#export PATH=/soft/compilers/intel-2019/compilers_and_libraries/linux/bin/intel64/:$PATH
export OMPI_FC=ifort
export OMPI_MPICXX=dpcpp  

export LD_LIBRARY_PATH=/home/bhomerding/install/proj-5.2.0/lib:$LD_LIBRARY_PATH
