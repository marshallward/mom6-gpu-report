# ISSUES 

Contents:

- High level TODOs
- MOM6-examples known issues on GPUs
- MPI related issues
- NVHPC versions issues
- How to compile NetCDF using CMake


## High level TODOs

- MOM6 examples seems to struggle with FMS2
- MOM6 examples does not use latest FMS2
- FMS2 latest CMake build system does not have a way to build the unit tests
- FMS2 autoconf build fails lots of unit tests when compiling with NVHPC

## MOM6-examples issues on GPUs 

The version of FMS2 that is the default in MOM6 examples seems to be giving problems. You have to switch back to FMS1 by modifying the Makefile in `ocean_only` to point to FMS1: `FMS_CODEBASE ?= ../src/FMS1` 

### FMS2 detailed issues (to be added)


## MPI related 

```
Authorization required, but no authorization protocol specified
Authorization required, but no authorization protocol specified
NOTE: MPP_DOMAINS_SET_STACK_SIZE: stack size set to    32768.
 &MPP_IO_NML
 HEADER_BUFFER_VAL =        16384,
 GLOBAL_FIELD_ON_ROOT_PE =  T,
 IO_CLOCKS_ON =  F,
 SHUFFLE =            0,
 DEFLATE_LEVEL =           -1,
 CF_COMPLIANCE =  F
 /
NOTE: MPP_IO_SET_STACK_SIZE: stack size set to     131072.
NOTE: ======== Model being driven by MOM_driver ========
NOTE: callTree: o Program MOM_main, MOM_driver.F90

FATAL: fms_affinity_set: OCEAN cpu_set size > allocated storage


FATAL: fms_affinity_set: OCEAN cpu_set size > allocated storage

--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
with errorcode 1.

NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
You may or may not see output from other processes, depending on
exactly when Open MPI kills them.
--------------------------------------------------------------------------
```

I've seen this error above when compiling with OpenMPI 5.0.5 on Gadi and both 4.1.4 and 5.0.5 on a personal machine. The solution was to run the MOM6 executable as `mpirun -np 1 ../build/MOM6` and ta-da

### Tested MPI versions 

- Openmpi
  - 4.1.4
  - 4.1.7
  - 5.0.5 

## NVHPC Version related 

### Compute sanitizer and opt flags 

Apparently using `-g` does not just add debug symbols but it might also hinder optimization. We have to use `-gopt`

### 25.5 

On 25.5 building and running the `double-gyre` example works well. However, running the `benchmark` example produces mapping errors on the GPU values. Known bug, see [here](https://forums.developer.nvidia.com/t/bug-nvhpc-25-x-present-table-errors-with-fortran-do-concurrent-and-kind-of-nested-type-bound-procedures/333144)

### 24.9 

Both double gyre and benchmark seem to work well. 

## Compiling your own stuff 

*attention* netcdf built with 25.5 won't be picked up by MOM if building MOM with nvhpc 24.9. You'll need versioned installs. sorry

```
git clone git@github.com:Unidata/netcdf-fortran.git
git clone git@github.com:Unidata/netcdf-c.git
export nvhpc_verno=25.5
export NETCDFC_ROOT=$PSCRATCH/install/nvfortran/${nvhpc_verno}/netcdf-c
export NETCDFF_ROOT=$PSCRATCH/install/nvfortran/${nvhpc_verno}/netcdf-fortran
cd netcdf-c
mkdir build
cd build 
cmake -DCMAKE_INSTALL_PREFIX=$NETCDFC_ROOT -DCMAKE_BUILD_TYPE=Release ../
make -j install
cd ../../netcdf-fortran 
mkdir build
cd build
	cmake -DCMAKE_INSTALL_PREFIX=$NETCDFF_ROOT -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$NETCDFC_ROOT ../
make -j install
```

Then...

```
#!/bin/bash

export nvhpc_verno=25.5
export NETCDFC_ROOT=$PSCRATCH/install/nvfortran/${nvhpc_verno}/netcdf-c
export NETCDFF_ROOT=$PSCRATCH/install/nvfortran/${nvhpc_verno}/netcdf-fortran

export PATH=$PATH:$NETCDFC_ROOT/bin
export PATH=$PATH:$NETCDFF_ROOT/bin
# check if your netcdf is pointing to lib64 or lib could be different
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NETCDFC_ROOT/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NETCDFF_ROOT/lib64
```

Now you can source the above file and have netcdf loaded, yay. 
