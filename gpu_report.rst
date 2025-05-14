======================
MOM6 GPU Porting Guide
======================

:author: Marshall Ward, Edward Yang
:geometry: margin=3cm

.. TODO: Cannot split author into a list?

This document describes the current state of porting MOM6 to GPU platforms.  We
hope to provide essential context on the following questions.

* Why should MOM6 be ported to GPUs?

  * Will we always have access to this hardware?
  * Could it become the *only* viable option for massively parallel jobs?
  * Will we maintain support on CPU?  Could it become "legacy" hardware?

* What are the available technologies?  How suitable is each technology for
  MOM6?  Some issues to consider:

  * Can it faithfully represent the MOM6 algorithms?
  * Is the code maintainable?  Or even readable?
  * Do we have explict control over the implementation?
  * Does it depend on a particular software stack?  Is it a sprawling mess?
  * Can we preserve the existing codebase?
  * Is the new implementation bit-equivalent to the CPU result?

We will try to address the questions above (over time, most likely).  For now,
our conclusion is to preserve the existing codebase and introduce OpenMP
directives.

The latter part of this document is intended to be a "how-to" for new users.
We hope to include the following:

* Instructions for compiling models with GPU support.  We currently focus on
  Nvidia, but should consider other platforms (AMD, possibly Intel?).

* Tutorial for using OpenMP for GPU migration, with a focus on MOM6.

* Relative performance of different strategies.

This is a work in progress.  Not every section is well-structured or even in a
readable form.  At this early stage, our goal is to accumulate information.  In
other words, please excuse the mess.


Motivation
==========

Utilizing new platforms

  GPUs represent a new compute device which we currently cannot use.  They are
  becoming widespread in HPC systems, and we are denying such users the ability
  to run MOM6.

Performance improvements

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

Directive-based Methods
-----------------------

Directives are incorporated into an existing language (C, C++, Fortran).  This
allows an existing codebase to be compiled to CPU and GPU targets.

.. code:: fortran

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
primarily designed for particular platform, but they also aspire for some
generality and there is some limited cross-platform support.

Program loops are written as kernels in the native language, which can be
compiled into GPU bytecode.  These kernels are interfaced to a higher level
language such as C.

.. code:: cpp

   // Define the kernel
   __global__ void add(float *a, float *b, float *c, float *d) {
      if (blockIdx.x < N)
         a[i] = b[i] * c[i] + d[i];
      }
   }

   // Call the kernel from main()
   vector_add<<n,1>>(a, b, c, d, n);

Languages like CUDA are not compiled into CPU bytecode, so a separate CPU
implementation loop may be required for cross-platform support.


Other options
-------------

Higher level libraries try to incorporate multiple kernel DSLs into a generic
framework.  Examples include Kokkos, OpenCL, and SYCL.  These are all C++
libraries, but could presumably be interfaced to other languages like Fortran.

Other languages try to entirely abstract the GPU interface.  Python and Julia
have extensive APIs into various GPU kernel framework.

.. TODO examples?


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

We are currently testing on an Ampere A100 GPU.

.. code::

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


Enabling OpenMP Offloading
--------------------------

I am currently using the following flags.

.. code:: make

   FCFLAGS += -mp=gpu -Mnofma -Minfo=all
   LDFLAGS += -mp=gpu

``-mp=gpu``
   This instructs the compiler to convert OpenMP directives to GPU bytecode.

  Both compiler and linker require ``-mp=gpu``.  Internally, the flag is used to
  access CUDA libraries.

``-Mnofma``
   This disables FMAs in the bytecode output, in both CPU and GPU.

   MOM6 provides parentheses to disable FMAs in critical regions, allowing us
   to safely use them where appropriate.  This appears to be respected in the
   CPU output, but the GPU output does not seem to respect parentheses when
   producing FMA output, and we see answer changes.

   Until this is fixed on the compiler side, we much for now disable FMAs.

``-Minfo=all``
   Ths is not necessary, but provides interesting (if overwhelming) updates on
   GPU usage.

.. TODO: Error for missing LDFLAGS?


``do concurrent`` Support
-------------------------

``-stdpar=gpu``
   Migrate ``do concurrent`` loops to GPU.

   This appears to generate an independent kernel for the loop, which is run on
   the GPU.  Managed memory is the default configuration, but this appears to
   run extremely slow for us.  When using do-concurrent, we want to disable
   managed memory.

``-gpu=nomanaged``
   Disable managed memory and explicitly move arrays. Recent versions of NVHPC
   will prefer ``-gpu=mem:separate``.

   Despite being a major burden for the developer, this has so far proven to be
   the best option for us.

   We should continue to explore the viability of managed or unified memory,
   but so far the timings of managed memory have been over 5x slower.


Non-Nvidia devices and Compilers
--------------------------------

I have not yet done any testing on AMD or Intel GPUs.  Consider this a
placeholder for future documentation.


Testing in MOM6
===============

Compiling
---------

Current testing is restricted to the ocean-only driver.  The MOM6-examples
repository includes a Makefile for building the executable.

.. code:: sh

   $ git clone https://github.com/NOAA-GFDL/MOM6-examples.git --recursive
   $ cd MOM6-examples/ocean_only
   $ CC=nvcc \
     FC=nvfortran \
     FCFLAGS="-g -O0 -mp=gpu -stdpar=gpu -Mnofma -Minfo=all" \
     LDFLAGS="-mp=gpu" \
     make -j

(Not yet tested... but you get the idea.)


Procedure
---------

Running and testing the code changes is still a work in progress.  The current
procedure is very simple and somewhat ad-hoc.  I will describe below my
process.

1. Compile the CPU and GPU executables.  Aside from GPU flags, all others
   should be identical.

   Currently I use the MOM6-examples ``ocean_only`` Makefile.  (Details to be
   added.)

2. Run the ``double_gyre`` test.  Verify no runtime errors.

   This is a layered test with no thermodynamics and modest physics.  **Porting
   this test to GPU is our first milestone.**

   Often the model will quickly go unstable and fail if something was not
   correctly transferred.

3. Verify equivalence of ``ocean.stats`` from CPU and GPU runs.

   We are prepared to relax this requirement if necessary.  But so far this
   equivalance has held, and we don't want to give it up lightly.

4. Repeat with ``benchmark``.  This a flexible test which strongly resembles
   past production runs.  It includes thermodynamics.  At a minimum, we
   want to ensure that our changes do not break this run.  Ideally, we would
   like to also move the thermodynamics onto the GPU.  (But see "Known
   Issues".)

At some point, we should extend our CI testing to GPUs, but this has proven to
be a decent procedure for exploring OpenMP capability.


MOM6 Directive Implementation
=============================

This section will try to summarize what we have learned so far about GPU
development and how to apply it to MOM6.  This is a summary of techniques --
and mistakes -- that we have learned on the way.

Our first goal is to try and migrate the dynamic core of the model.  We
specifically focus on the split timestep RK2 solver,
``MOM_dynamics_split_RK2.F90``.  We aspire for bitwise identical answers with
the CPU solution.

Ideally, the fields associated with the dynamic core should remain on the GPU
for the entire run.  But the work will have to be done in pieces, often one
loop at a time.


Loop migration
--------------

The main task is to accumulate loops into GPU kernels for migration.   Each
kernel is bounded by ``$!omp target`` directives.

The following creates one GPU kernel with one serial loop (``k``) and two
parallelized loops (``i``, ``j``).

.. code:: fortran

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

Kernel is bounded by ``!$omp target`` ... ``!$omp end target``.  This defines a
unit of execution on the GPU.  A kernel can contain multiple loops.

``collapse(N)`` tells it to merge the nested loop into a single large loop.
This can presumably avoid pipelining issues across dimensions.  For now, this
should be considered an optimization and not required for porting.


The ``!$omp parallel loop`` Directive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This directive is a relatively new addition to OpenMP.  It can be considered
shorthand for the following directive::

   !omp teams distribute parallel do simd

``teams`` are collections of threads with shared resources.  In an Nvidia GPU,
the teams are SM processors, and loops is parallelized over the threads of the
SM processor.

A possibly faster form of the previous loop is shown below.

.. code:: fortran

   !$omp target
   !$omp teams distribute
   do k=1,nz
     !$omp parallel do collapse(2)
     do j=js,je ; do I=Isq,Ieq
       u_bc_accel(I,j,k) = (CS%CAu_pred(I,j,k) + CS%PFu(I,j,k)) + CS%diffu(I,j,k)
     enddo ; enddo

     !$omp parallel do collapse(2)
     do J=Jsq,Jeq ; do i=is,ie
       v_bc_accel(i,J,k) = (CS%CAv_pred(i,J,k) + CS%PFv(i,J,k)) + CS%diffv(i,J,k)
     enddo ; enddo
   enddo
   !$omp end target

The ``simd`` directs the team to use SIMD-like instructions over the threads.
This is almost always the default behavior, so it is often omitted.

Note that as of `9th April, 2025` AMD compilers don't understand the `loop`
directive.


Data Migration
--------------

We should aim minimize data transfer between the CPU host and GPU target.  This
is achieved by keeping the arrays on the GPU across multiple kernels.

Data directives are used to move an array between host and target.  This
operations occur outside of any compute kernels.

To move an array from host to device, or vice versa::

   !$omp target enter data map(to: x)

This allocates a new ``x`` on the GPU and sets the values from the CPU. If ``x``
has been mapped to the GPU previously (either through ``map(to: ...)`` or
``map(from: ...)``), then this increments the reference counter to ``x`` by 1,
but leaves ``x`` unchanged (see `OpenMP 5.0 API Spec`_ pg 318 line 18 to pg 
319 line 9).

To move data from GPU back to CPU::

   !$omp target exit data map(from: x)

This decrements the reference counter to ``x`` by 1, and **only transfers data
back to the CPU if the reference counter hits 0** (`OpenMP 5.0 API Spec`_ pg
319 line 18 to line 19).

Arrays can be independently allocated or deleted on the GPU.  This block
allocates ``h`` on the GPU but does not fill its data. If ``h`` is already
present, the reference counter is incremented.

.. code:: fortran

   allocate(CS%h(isd:ied,jsd:jed,nz))
   CS%h(:,:,:) = GV%Angstrom_H
   !$omp target enter data map(alloc: CS%h)

This block sets the reference counter of ``h`` to zero and deallocates ``h``
on the GPU (`OpenMP 5.0 API Spec`_ pg 319 lines 27-28).

.. code:: fortran

   !$omp target exit data map(delete: h)

This block decrements ``h`` reference counter by 1. If the reference counter
hits zero, then ``h`` is also deallocated on the GPU (`OpenMP 5.0 API Spec`_
pg 319 lines 24-26).

.. code:: fortran

    !$omp target exit data map(release: h)

If you want to exchange values between a array which already exists on the GPU,
use ``update``.

.. code:: fortran

  !$omp target update to(h)
  call PressureForce(h, tv, CS%PFu, CS%PFv, G, GV, US, CS%PressureForce_CSp, &
                     CS%ALE_CSp, p_surf, CS%pbce, CS%eta_PF)
  !$omp target update from(CS%PFu, CS%PFv, CS%pbce, CS%eta_PF)

The ``to`` and ``from`` modifiers are with respect to the target GPU.

.. OpenMP has a ``present()`` modifier to explicitly declare that an array is
   already on the target GPU.  But most compilers still do not support this
   modifier.  In Nvidia, the runtime appears to handle this well and avoids
   redundant transfers, so it is probably not necessary to use ``present()``.
   But this is still something that should be monitored closely.

.. _OpenMP 5.0 API Spec: https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5.0.pdf


Scalar data transfer
~~~~~~~~~~~~~~~~~~~~

OpenMP will automatically identify and transfer any scalar data between host
and target, so these can be omitted from data transfer directives.


Derived type transfers
~~~~~~~~~~~~~~~~~~~~~~

Derived types should be explicitly transferred to the GPU.  If the derived
type contains any allocatable arrays, then these must also be separately
allocated and transferred.

The example below shows the data transfer of the MOM6 grid object and some of
its arrays.

.. code:: fortran

   !$omp target enter data map(to: G)
   !$omp target enter data map(to: G%dxCu, G%dyCv)
   !$omp target enter data map(to: G%IdxCu, G%IdyCv)
   !$omp target enter data map(to: G%mask2dBu, G%mask2dT)


Partial Data Transfer
~~~~~~~~~~~~~~~~~~~~~

In Fortran, a data transfer will copy the entire array between host and target
if the index bounds are omitted.  This is an advantage over C and C++, whose
arrays use pointer-based allocation and their size must be independently
tracked.

When necessary, it is possible to restrict transfer to an array slice.  The
example below adjusts the bottom layer to account for self-attraction and
loading.

.. code:: fortran

  !$omp target update from(e(:,:,nz+1))
  call calc_SAL(SSH, e_sal, G, CS%SAL_CSp, tmp_scale=US%Z_to_m)
  do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
    e(i,j,nz+1) = e(i,j,nz+1) - e_sal(i,j)
  enddo ; enddo
  !$omp target update to(e(:,:,nz+1))

However, be careful with arrays with rank 3 and above! Consider the below
declaration and subsequent data transfer:

.. code:: fortran

   real:: a(10, 20, 30)

   !$omp target enter data map(to: a(3:8, 3:18, :))
   ... do work ...
   !$omp target exit data map(from: a(3:8, 3:18, :))

Both the ``enter`` and ``exit`` statements trigger ``(18-3+1)*30 = 480``
transfers of ``4*(8-3+1) = 24`` bytes of data to/from the GPU! So, depending on
the size/number of slices, it may be better to send more data than you need.
For some reason, ``map(to: a(3:8, :, :))`` triggers only one transfer.

I'm not sure of the exact reason why this happens!

Data regions
------------

An array can be defined to exist within a particular region.  The example below
uses the temporary array ``dM`` when applying a reduced gravity adjustment to
the pressure force.

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
define the scope of a variable.  (TODO: Show a more complex example.)


OpenMP Targets and MPI (WIP!)
-----------------------------

OpenMP should support passing data on device to MPI calls by using data regions
and ``use_device_ptr`` or ``use_device_addr``. Doing so should allow for direct
GPU to GPU data transfers, assuming the MPI library was built with relevant GPU
support. See `working example from AMD`_.

.. _working example from AMD: https://github.com/FluidNumerics/gpu-programming/blob/main/samples/fortran/mpi%2Bopenmp/main.f90

.. code:: fortran

   ! make sure data is initalized on device
   !$omp target enter data map(to: a)
   ! ... do stuff ...
   ! initialize new data region, and make sure device data is used
   !$omp target data use_device_ptr(a) ! newer compilers might prefer use_device_addr
   if (rank == 0) call MPI_Send(a, ...)
   if (rank == 1) call MPI_Recv(a, ...)
   !$omp end target data
   !$omp target exit data map(from: a)


Pseudo-profiling for tracking data transfers
--------------------------------------------

``NV_ACC_NOTIFY`` is an undocumented environment variable that you can set to
monitor data transfers triggered by the OpenMP/OpenACC runtime.

Settings are configured by bitmasked values.

* 1: kernel launches

* 2: data transfers

* 4: wait operations or synchronizations with the device

* 8: region entry/exit

* 16: data allocate/free

For example, ``NV_ACC_NOTIFY=2 ../build/MOM6`` will dump a bunch of information
to your terminal like:

.. code::

   upload CUDA data  file=<src-file> function=zonal_mass_flux line=617 device=0
       threadid=1 variable=dt bytes=8
   upload CUDA data  file=<src-file> function=zonal_mass_flux line=617 device=0
        threadid=1 variable=h_in(ish-1:ieh,:,:) bytes=34560
   upload CUDA data  file=<src-file> function=zonal_mass_flux line=617 device=0
        threadid=1 variable=h_w(ish-1:ieh,:,:) bytes=34560
   upload CUDA data  file=<src-file> function=zonal_mass_flux line=617 device=0
        threadid=1 variable=h_e(ish-1:ieh,:,:) bytes=34560
   upload CUDA data  file=<src-file> function=zonal_mass_flux line=617 device=0
        threadid=1 variable=g bytes=12808
   ... a lot more

The information can be manipulated to find where your transfers are happening.
For example, you're porting a subroutine and want to find what transfers are
happening in that subroutine:

.. code:: bash

   NV_ACC_NOTIFY=2 ../build/MOM6 2>&1 > mom6-dump.txt
   grep zonal_flux_layer | sort mom6-dump.txt | uniq -c | sort -n

Which yields the number of transfers for a particular variable in ascending
order:

.. code::

   ... a lot more lines
    80356 upload CUDA data  file=/.../MOM_continuity_PPM.F90 function=merid_flux_layer
          line=2061 device=0 threadid=1 variable=h_s(ish:ieh,j:j+1) bytes=704
    80356 upload CUDA data  file=/.../MOM_continuity_PPM.F90 function=merid_flux_layer
          line=2061 device=0 threadid=1 variable=por_face_areav(ish:ieh,j) bytes=352
    80356 upload CUDA data  file=/.../MOM_continuity_PPM.F90 function=merid_flux_layer
          line=2061 device=0 threadid=1 variable=visc_rem(ish:ieh) bytes=352
   213036 upload CUDA data  file=/.../MOM_continuity_PPM.F90 function=merid_flux_layer
          line=2061 device=0 threadid=1 variable=.attach. bytes=200
   213036 upload CUDA data  file=/.../MOM_continuity_PPM.F90 function=merid_flux_layer
          line=2093 device=0 threadid=1 variable=.detach. bytes=8

This information you can use to target variables to map in data regions or when
using ``enter/exit`` statements. Additionally, you can `wc -l mom6-dump.txt`
before and after to see whether your changes successfully reduced the number of
transfers.

NB: ``NV_ACC_NOTIFY=3`` tells you kernel launch information.


Data management across files
----------------------------

MOM6 variables are defined over multiple files, and we need to ensure that
there are no unnecessary data transfers as data is moved across functions of
different translation units.

There is no restriction to allocating and transferring an array in one file and
using the array in a kernel defined in another file.  The compiler appears to
correctly track the array address across files.  However, the user must be
careful to ensure that the arrays exist, or errors will be raised.  (Usually a
"partially present" error.)


Procedure calls
---------------

Procedures can be compiled to GPU bytecode with ``!$omp declare target``.

.. code:: fortran

   function cuberoot(x)
      real, intent(in) :: x
      real :: cuberoot
      !$omp declare target

      cuberoot = x**(1./3.)
   end subroutine

This allows the procedure call to reside within a kernel, or even within a
loop.

.. code:: fortran

   !$omp target
   !$omp parallel loop
   do i = 1, N
      r(i) = cuberoot(u(i))
   enddo
   !$omp end target

(TODO: Find an in-code example)

This has not been very useful in practice.  A procedure can only be compiled if
its entire contents can be run on the GPU, and we still encounter a lot of
constructs which do not work.


Known Issues
============

TODO

* Procedure pointers

* Type-bound procedures (both static and virtual functions)

* Complex derived types (esp. the open boundary conditions)

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

Some success has been had with ``nsys profile -t openacc --stats=true``, as it
collects both CUDA API calls and OpenACC regions (NVIDIA compilers maps OpenMP
constructs to OpenACC ones).


Common Errors
-------------

Sadly, most errors are either generic

* (Runtime) "partially present"

  Typically this means that an array is not on the device or an allocated array
  or array section hasn't been freed e.g. in an exit data statement.

*


Memory monitoring
-----------------

We need some tooling here.  We have no idea how memory is being used.  CUDA
memory?  Unified memory?  In-chip?  (probably not).

Most likely we are not using our memory well.


Miscellaneous
=============

CPU parallelization
-------------------

Very basic testing suggests that we can replace existing OpenMP directives with
the newer target-based directives.

For the Nvidia compiler, using either ``-mp=multicore`` or ``-mp=autopar`` will
distribute the loop over multiple threads.  But this has not been tested in
production and needs more investigation.

There is also no guarantee that this will work in other compilers.


Compiler support
----------------

NVHPC Optimization parser problems?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If I do an ``-O2`` CPU build with 24.5, then something strange happens with the
parser.::

   FATAL: MOM_file_parser : the parameter name 'DT' was found without define or undef. Line: 'DT =' in file MOM_input.

The file parser sets the last character (``last``) to 4, even though it's
looking for ``" = "``.  If I change the search to ``=``, then I get a new
error::

   FATAL: read_param_real: Variable DT found but not set in input files.

Oddly, none of this happens in the GPU build, even though there are no GPU
structures in the file.

Not sure what's going on, but it's preventing ``-O2`` comparisons between CPU
and GPU.

(This could be fixed in 24.9 or 25.x, needs testing)


OpenMP handler problems
~~~~~~~~~~~~~~~~~~~~~~~

The OpenMP-based version of ``set_pbce_Bouss`` chokes on newer compilers (24.9,
25.1).  It works with 24.5.  This is a compile-time error, not runtime.

The problem doesn't appear to be with any particular loop.  Nor does it
necessarily be caused by the ``target data`` regions.  However, it is one of
the few regions using ``target data``, so it could be related.

Still looking into this one.


NVHPC 25.1 flag mismatch
~~~~~~~~~~~~~~~~~~~~~~~~

nvhpc 25.1 struggles with certain flag combinations on certain files.

If I use ``nvfortran -O2 -O0`` on FMS ``diag_data.F90``, then I see an error::

   $ nvfortran -c -O2 -O0 test.f90
   nvfortran-Info-Switch -Mvect forces -O2
   NVFORTRAN-F-0000-Internal compiler error. Deferred-length character symbol must have descriptor     730  (test.f90: 27)
   NVFORTRAN/x86-64 Linux 25.1-0: compilation aborted

It seems to come from the ``-Mvect`` flag::

   $ nvfortran -c -Mvect test.f90
   NVFORTRAN-F-0000-Internal compiler error. Deferred-length character symbol must have descriptor     730  (test.f90: 27)
   NVFORTRAN/x86-64 Linux 25.1-0: compilation aborted

Apparently it's some config problem on their end.  This seems to magically
fix the problem::

   $ nvfortran -c -O2 -O0 -Hx,53,2 test.f90
   nvfortran-Info-Switch -Mvect forces -O2

This was discovered because ``mpifort`` always used ``-O2`` on one of our
machines.


GCC issues
~~~~~~~~~~

OpenMP offloading to target GPUs is a relatively new feature.  This was
introduced in OpenMP 4.0, and didn't quite catch up to OpenACC until 5.x.

*Our GCC tests in GitHub Actions cannot compile these tests!*

.. code::

   /home/runner/work/MOM6/MOM6/src/core/MOM_PressureForce_FV.F90:1687:18:

    1687 |   !$omp   map(to: tv_tmp, tv_tmp%T, tv_tmp%S, tv, tv%eqn_of_state, EOSdom2d)
         |                  1
   Error: List item ‘tv_tmp’ with allocatable components is not permitted in map clause at (1)

Allocatables in derived types were added in 5.0 and is still not supported in
GCC 14.

https://gcc.gnu.org/onlinedocs/gcc-13.1.0/libgomp/OpenMP-5_002e0.html





Loop dependencies within a kernel
---------------------------------

It is still not clear to me when loop dependencies can be managed within a
kernel.  For example, ``gradKE()`` in ``MOM_CoriolisAdv.F90``.

.. code:: fortran

   !$omp target
   if (CS%KE_Scheme == KE_ARAKAWA) then
     !$omp parallel loop collapse(2)
     do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
       KE(i,j) = ( ( (G%areaCu( I ,j)*(u( I ,j,k)*u( I ,j,k))) + &
                     (G%areaCu(I-1,j)*(u(I-1,j,k)*u(I-1,j,k))) ) + &
                   ( (G%areaCv(i, J )*(v(i, J ,k)*v(i, J ,k))) + &
                     (G%areaCv(i,J-1)*(v(i,J-1,k)*v(i,J-1,k))) ) )*0.25*G%IareaT(i,j)
     enddo ; enddo
   elseif (CS%KE_Scheme == KE_SIMPLE_GUDONOV) then
     ! ...
   endif

   !*** Split the kernel here??

   ! These loops depend on KE(:,:)

   !$omp parallel loop collapse(2)
   do j=js,je ; do I=Isq,Ieq
     KEx(I,j) = (KE(i+1,j) - KE(i,j)) * G%IdxCu(I,j)
   enddo ; enddo

   ! Term - d(KE)/dy.
   !$omp parallel loop collapse(2)
   do J=Jsq,Jeq ; do i=is,ie
     KEy(i,J) = (KE(i,j+1) - KE(i,j)) * G%IdyCv(i,J)
   enddo ; enddo
   !$omp end target

If I do not split the ``KE`` from ``KE[xy]``, then there are errors in some
experiments.  I can't definitively blame this on concurrently or
parallelization, but there are numerical errors.  Splitting the kernels
restores the solution.

This is not the only instance of data dependencies across loops within a
kernel.  Yet this is the example which chokes.

* Do I need to somehow express this dependency in the ``parallel loop``
  directive?

* Am I *supposed* to split the kernel?  Is that the correct move?

Feedback and/or futher study is needed here.  (Maybe even just a read of the
OpenMP standard?)


Redundant target update
-----------------------

Certain loops on GPU currently fail to reproduce the CPU numbers unless
redundant ``!$omp target update`` is appled.  For example, see
``MOM_hor_visc.F90``.

.. code:: fortran

   !$omp target enter data map(to: u)
   ! ...
   do k = 1, n
     !$omp target
     !$omp parallel loop collapse(2)
     do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
       dudx(i,j) = CS%DY_dxT(i,j)*((G%IdyCu(I,j) * u(I,j,k)) - &
                                   (G%IdyCu(I-1,j) * u(I-1,j,k)))
     enddo ; enddo
     !$omp end target
   enddo

Even though ``u`` has been updated to GPU, it appears to be using somewhat
outdated values.  If an additional ``update`` directive is applied,

.. code:: fortran

   !$omp target enter data map(to: u)
   ! ...
   do k = 1, n
     !$omp target update to(u(:,:,k))

     !$omp target
     !$omp parallel loop collapse(2)
     do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
       dudx(i,j) = CS%DY_dxT(i,j)*((G%IdyCu(I,j) * u(I,j,k)) - &
                                   (G%IdyCu(I-1,j) * u(I-1,j,k)))
     enddo ; enddo
     !$omp end target
   enddo

then CPU-GPU equivalence is restored.

There are other instances of this problem in the model (e.g. continuity
solver).  Is this a compiler bug?  Or an error in the code directives?


Nested and cross-subroutine parallelism
---------------------------------------

In MOM6, it's not uncommon to have large 3D loops written such that the
outer-most loop encompass many nested inner-loops, where outer loop
iterations are independent. A hopefully easy-to-read example of this 
is in `horizontal_viscosity`_, which also `calls subroutines`_
that perform the inner loops. You may not want to refactor, so instead
you could try leverage nested parallelism.

.. _horizontal_viscosity: https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/parameterizations/lateral/MOM_hor_visc.F90#L702
.. _calls subroutines: https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/parameterizations/lateral/MOM_hor_visc.F90#L1085

For the case where the outer loops contain **multiple independent** inner 
loops, you can distribute the outer loop across OpenMP target teams. The
inner loops can then be parallelised within each team. Below is a 
contrived example:

.. code:: fortran

   !$omp target teams distribute private(x, y) map(from: z)
   do k = 1, nz

      !$omp parallel do simd collapse(2)
      do j = 1, nj
         do i = 1, ni
            x(i, j) = ...
            y(i, j) = ...
         enddo
      enddo

      ... maybe more similar loops ...

      !$omp parallel do simd collapse(2)
      do j = 1, nj
         do i = 1, ni
            z(i, j, k) = x(i, j) + y(i, j)
         enddo
      enddo
   enddo ! end of k-loop


The first directive creates a private copy of ``x`` and ``y`` in each team.
If the array size isn't known at compile time, ``nvfortran`` seems to assume
that the private array is small, and will try to allocate space on
shared memory (memory visible within the team, and faster than global).

If the inner loops are in another subroutine, the ``!$omp declare target``
subroutine can be utilized:

.. code:: fortran

   subroutine do_something(ni, nj, in_array, out_array)
      implicit none
      integer, intent(in):: ni, nj
      real, intent(in):: in_array(ni, nj)
      real, intent(out):: out_array(ni, nj)
      real:: tmp_array(ni, nj)
      integer:: i, j

      ! tell the compiler that the subroutine will be called in a
      ! target region.
      !$omp declare target

      ! put parallel do inside
      ! nb I run into errors when using collapse inside
      !$omp parallel do simd
      do j = 1, nj
         do i = 1, ni
            tmp_array(i, j) = ...
         enddo
      enddo

      !$omp parallel do simd
      do j = 1, nj
         do i = 1, ni
            out_array(i, j) = ...
         enddo
      enddo

   end subroutine do_something

   ! ... in the main loop
   !$omp teams distribute ...
   do k = 1, nz
      !$omp parallel do simd collapse(2)
      do j = 1, nj
         do i = 1, ni
            ! ... do something ...
         enddo
      enddo

      call do_something(...)
   enddo


Problems with nested parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Shared memory is limited on GPUs (24-48kB per block/team for NVIDIA). 
  Exceeding shared memory will degrade performance as arrays go into global memory.
* GPU static memory is limited, so if you jump into a subroutine that allocates
  lots of static arrays, it doesn't take much to OOM (see relevant `NVIDIA forum post`_).
* Jumping into a target subroutine segfaults when an argument is a pointer.
* I get incorrect results when the ``parallel do`` inside a target subroutine is
  coupled with ``collapse()``.
* I've found that explicit nested parallelism performs meaningfully worse than
  refactoring the loops into separate ``kji`` blocks.

.. _NVIDIA forum post: https://forums.developer.nvidia.com/t/issue-with-automatic-array-in-device-subroutine-defined-with-openacc-directive/245873/2
