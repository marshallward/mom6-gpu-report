===========================
MOM6 GPU activities at GFDL
===========================

Summary of GPU-related activities around MOM6 at GFDL.


Motivation
==========

* Utilizing new platforms

  GPUs represent a new compute device which we currently cannot use.  They are
  becoming widespread in HPC systems, and we are denying such users the ability
  to run MOM6.

* Performance improvements

  GPUs can provide higher throughput in certain situations.  A typical GPU
  device has a greater number of compute cores, and a job may run faster if its
  parallelization is greater than other costs such as reduced CPU speeds.

  For memory-bound codes such as MOM6, we can benefit from the greater RAM
  speeds.

  In both cases, a 5-10x speedup on each node is a reasonable expectation.


Development Options
===================

There are several options for compiling general-purpose source code into
proprietary GPU bytecode.

* Directives: OpenMP, OpenACC

Directives are incorporated into an existing language (C, C++, Fortran).  This
allows an existing codebase to be compiled to CPU and GPU targets.

.. code::

   !$omp target
   !$omp parallel loop
   do i = 1, N
      a(i) = b(i) * c(i) + d(i)
   enddo
   !$omp end target

Directive support is not universal across vendor compilers.  It is not always
possible to tune a loop to a particular device.  If there are too many
directives, then it can harm the readability of the source code.


Kernel DSLs
-----------

CUDA and HIP are languages which target compilation to GPU bytecode.  They are
primarily used to target Nvidia and AMD devices, respectively, but there is
some limited cross-platform support.

Program loops are written as kernels in the native language, which can be
compiled into GPU bytecode.  These kernels are interfaced to a higher level
language such as C.

.. code::

   // Defining the kernel
   __global__ void add(float *a, float *b, float *c, float *d) {
      if (blockIdx.x < N)
         a[i] = b[i] * c[i] + d[i];
      }
   }

   // Calling the kernel from main()
   vector_add<<n,1>>(a, b, c, d, n);

Languages like CUDA cannot compiled into CPU bytecode, so multiple
implementations of a loop may be required.


Other options
-------------

Higher level libraries try to incorporate multiple kernel DSLs into a generic
framework.  Examples include Kokkos, OpenCL, and SYCL.  These are all C++
libraries, but could presumably be interfaced to other languages like Fortran.

Other languages try to entirely abstract the GPU interface.  Python and Julia
have extensive APIs into various GPU kernel framework.


Implementation in MOM6
----------------------

Most of the options above require extensi



OpenMP directives in MOM6
=========================
