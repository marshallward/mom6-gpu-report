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

Our first priority is to run the MOM6 solvers on both CPU and GPU platforms.
At this early stage, we would prefer a solution which prioritizes CPU
performance over GPU.


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

MOM6 has opted to preserve its Fortran codebase and pursue a directive-based
approach.

Most of the options above require extensive rewrites in new languages, as
either a kernel-based DSL or a new high-level language.  The MOM6 codebase is
very large -- over 200k lines -- and is being used in many research and
forecasting systems.  The dynamic ALE vertical coordinate introduces solvers
which are untested in these frameworks.  Any rewrite will require an additional
infrastructure development, which will only increase the development cost.

We are currently pursuing OpenMP directives.  OpenMP is a platform-independent
language which is supported by all vendor compilers.  The other option,
OpenACC, is primarily designed for Nvidia systems.  While there is limited
support for OpenACC in both GCC and AMD compilers, the Intel compilers
explicitly do not support OpenACC.


.. NOTE There are even reports that Nvidia compilers produce faster performance
   from OpenMP than its own OpenACC.  (Although I can't imagine why it would
   even differ...)


OpenMP support in MOM6
======================

System environment
------------------

Current testing is using the Nvidia's ``nvfortran`` compiler.::

  $ nvfortran --version

  nvfortran 24.5-0 64-bit target on x86-64 Linux -tp znver2

Nvidia is transitioning to a new LLVM-based ``flang`` compiler.  Future major
development efforts will be directed to ``flang``, including OpenMP support.

We are currently testing on an Ampere A100 GPU.::

   $ nvidia-smi
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
   |-----------------------------------------+------------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  NVIDIA A100-PCIE-40GB          On  |   00000000:25:00.0 Off |                    0 |
   | N/A   35C    P0             36W /  250W |       1MiB /  40960MiB |      0%      Default |
   |                                         |                        |             Disabled |
   +-----------------------------------------+------------------------+----------------------+

Some preliminary testing was done on a Volta V100, and we will soon have
access to Hopper H100s.  I would not expect much difference with respect to
compilation, but we may need to be aware of the respective memory on each
device.


Compilation instructions
------------------------

Compilation requires the following flags::

   FCFLAGS += -mp=gpu -Mnofma -Minfo=all
   LDFLAGS += -mp=gpu

``-mp=gpu`` enables GPU migration of OpenMP directives.  ``-Mnofma`` is used
for reproducibility, since Nvidia compilers ignore parentheses when applying
FMA instructions.  ``-Minfo`` is useful for monitoring GPU instructions.
although it can be a bit overwhelming.

Note that ``-mp=gpu`` needs to be applied during both compiling and linking.


Non-Nvidia devices
------------------

I have not yet done any testing on AMD or Intel GPUs.  Consider this a
placeholder for future documentation.


MOM6 Directive Implementation
=============================

This section will try to summarize what we have learned so far about GPU
development and how to apply it to MOM6.  This is a summary of techniques --
and mistakes -- that we have learned on the way.

Our first goal is to try and migrate the dynamic core of the model.  We
specifically focus on the split timestep RK2 solver,
``MOM_dynamics_split_RK2.F90``.  We should aspire for bitwise identical answers
with the CPU solution.

Ideally, the fields associated with the dynamic core should remain on the GPU
for the entire run.  But the work will have to be done in pieces, often one
loop at a time.


Loop migration
--------------

The main task is to accumulate loops into GPU kernels for migration.   Each
kernel is bounded by ``$!omp target`` directives.

The following creates one GPU kernel with one serial loop (``k``) and two
parallelized loops (``i``, ``j``).

.. code::

   !$omp target
   do k=1,nz
     !$omp parallel loop collapse(2)
     do j=js,je ; do I=Isq,Ieq
       u_bc_accel(I,j,k) = (CS%CAu_pred(I,j,k) + CS%PFu(I,j,k)) + CS%diffu(I,j,k)
     enddo ; enddo

     !$omp parallel loop collapse(2)
     do J=Jsq,Jeq ; do i=is,ie
       v_bc_accel(i,J,k) = (CS%CAv_pred(i,J,k) + CS%PFv(i,J,k)) + CS%diffv(i,J,k)
     enddo ; enddo
   enddo
   !$omp end target

Main notes:

* Kernel is bounded by ``!$omp target`` ... ``!$omp end target``.  This defines
  a unit of execution on the GPU.  A kernel can contain multiple loops.

* ``!$omp parallel loop`` directs a loop to be parallelized.  The compiler
  largely decides how to distribute this loop.

  Note that ``loop`` is a newer construct introduced to simplify
  parallelization.  Many existing online documentation uses a more explict form
  which separates the loop into ``teams``.  Something like this::

    !$omp target teams distribute parallel for

  but ``loop`` moves this fine-tuning to the compiler.  (At least that's my
  take on things.  Verify?)

* ``collapse(N)`` tells it to merge the nested loop into a single large loop.

  .. NOTE afaik N is the number of loops, but it could be N-1?


Data Migration
--------------

We should aim minimize data transfer between the CPU host and GPU target.  This
is achieved by keeping the arrays on the GPU across multiple kernels.

Data directives are used to move an array between host and target.  This
operations occur outside of any compute kernels.

To move an array from host to device, or vice versa::

   !$omp target enter data map(to: x)
   !$omp target exit data map(from: x)

``move(from: x)`` will deallocate ``x`` on the GPU after transferring its
values to the CPU.

.. But ``map(to: x)`` does not deallocate on the CPU, right?

Arrays can be allocated or deleted on the GPU, independent of the host::

   !$omp target enter data map(alloc: x)
   !$omp target exit data map(delete: x)

This can avoid unnecessary transfers.

If you want to transfer the *values* of an array, but keep them on the device,
use ``update``::

   !$omp target update to(x)
   !$omp target update from(x)

``to`` and ``from`` are with respect to the target GPU.

Note that OpenACC directives can use the ``present()`` modifier to explicitly
declare an array is on the GPU.


Data regions
------------

An array can be defined to only exist within a particular region.

.. code:: fortran

  !$omp target data map(alloc: dM)

  !$omp target
  !$omp parallel loop collapse(2)
  do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
    dM(i,j) = (CS%GFS_scale - 1.0) * (G_Rho0 * GV%Rlay(1)) * (e(i,j,1) - G%Z_ref)
  enddo ; enddo

  do k=1,nz
    !$omp parallel loop collapse(2)
    do j=js,je ; do I=Isq,Ieq
      PFu(I,j,k) = PFu(I,j,k) - (dM(i+1,j) - dM(i,j)) * G%IdxCu(I,j)
    enddo ; enddo
    !$omp parallel loop collapse(2)
    do J=Jsq,Jeq ; do i=is,ie
      PFv(i,J,k) = PFv(i,J,k) - (dM(i,j+1) - dM(i,j)) * G%IdyCv(i,J)
    enddo ; enddo
  enddo
  !$omp end target
  !$omp end target data

In this case the code can be further simplified by attaching the ``map()`` onto
the ``!$omp target`` directive.

.. code:: fortran

   !$omp target map(alloc: dM)
   ...
   !$omp end target

but for more complex blocks with multiple kernels, it can be a valuable way to
define the scope of a variable.


Data management across files
----------------------------

MOM6 variables are defined over multiple files, and we need to ensure that
there are no unnecessary data transfers as data is moved across functions of
different translation units.


Known Issues
============

TODO

* Function calls

* Complex derived types

* Excessive synchronization?

* ...?


Debugging and Profiling
=======================

The current state of both is very poor.  We need a lot of support here.

At the moment, I am relying mostly on ``nsys nvprof`` to get timing and data
transfer reports.

Nsight is obviously the way forward here, but there are some issues on my
systems's software stack which I have been unable to overcome.  (Could be me,
could be the system...)
