===========================
MOM6 GPU activities at GFDL
===========================

:author: Marshall Ward
:organization: NOAA-GFDL
:geometry: margin=3cm

Summary of GPU-related activities around MOM6 at GFDL.


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

``-mp=gpu`` enables GPU migration of OpenMP directives.  ``-Mnofma`` is used
for reproducibility, since Nvidia compilers ignore parentheses when applying
FMA instructions.  ``-Minfo`` is not required but is useful for monitoring GPU
instructions, although it can be a bit overwhelming.

Both compiler and linker require ``-mp=gpu``.  Internally, the flag is used to
access CUDA libraries.

.. TODO: Error for missing LDFLAGS?


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
       FCFLAGS="-g -O0 -mp=gpu -Mnofma -Minfo-all" \
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


MOM6 `double_gyre` GPU Porting Progress
=======================================

Below are a list of subroutines/functions that are used in the ``double_gyre``
test - sorted first by sources files which use up the most CPU time, then by the
subroutines/functions in those sources files which use up the most time.

.. code::

   - [ ] MOM_continuity_PPM.F90               1.207831s
      - [ ] meridional_flux_adjust            0.205482s - Edward
      - [ ] zonal_flux_adjust                 0.170399s
      - [ ] zonal_flux_layer                  0.095223s
      - [ ] zonal_flux_layer                  0.090211s
      - [ ] set_merid_bt_cont                 0.080188s
      - [ ] merid_flux_layer                  0.070164s
      - [ ] merid_flux_layer                  0.065153s
      - [ ] set_zonal_bt_cont                 0.060141s
      - [ ] ppm_reconstruction_x              0.045106s
      - [ ] zonal_mass_flux                   0.045106s
      - [ ] merid_flux_layer                  0.040094s
      - [ ] ppm_reconstruction_y              0.040094s
      - [ ] zonal_flux_thickness              0.040094s
      - [ ] merid_flux_layer                  0.035082s
      - [ ] meridional_mass_flux              0.030070s
      - [ ] ppm_limit_pos                     0.030070s
      - [ ] merid_flux_layer                  0.020047s
      - [ ] meridional_flux_thickness         0.020047s
      - [ ] ppm_limit_pos                     0.020047s
      - [ ] meridional_edge_thickness         0.005012s
      - [ ] continuity_merdional_convergence  0.0s
      - [ ] continuity_ppm                    0.0s
      - [ ] continuity_zonal_convergence      0.0s
      - [ ] zonal_edge_thickness              0.0s
   - [ ] MOM_barotropic.F90                   1.182772s
      - [ ] btstep                            1.002349s
      - [ ] set_local_bt_cont_types           0.080188s
      - [ ] find_uhbt                         0.040094s
      - [ ] find_vhbt                         0.040094s
      - [ ] bt_mass_source                    0.010023s
      - [ ] btcalc                            0.010023s
   - [ ] MOM_vert_friction.F90                0.726703s
      - [ ] vertvisc_coef                     0.355834s
      - [ ] vertvisc                          0.140329s
      - [ ] vertvisc_remnant                  0.120282s
      - [ ] find_coupling_coef                0.075176s
      - [ ] vertvisc_limit_vel                0.035082s
   - [ ] MOM_hor_visc.F90                     0.200470s
      - [ ] horizontal_viscosity              0.200470s
   - [ ] MOM_CoriolisAdv.F90                  0.125294s
      - [ ] coradcalc                         0.090211s
      - [ ] gradke                            0.035082s
   - [ ] diag_manager.F90                     0.070164s
      - [ ] send_data_3d                      0.070164s
   - [ ] MOM_set_viscosity.F90                0.055129s
      - [ ] set_viscous_bbl                   0.055129s
   - [ ] MOM_dynamics_split_RL2.F90           0.035082s
      - [ ] step_mom_dyn_split_rk2            0.030070s
      - [ ] register_restarts_dyn_split_rk2   0.005012s
   - [ ] MOM_PressureForce_FV.F90             0.025059s
      - [ ] pressureforce_fv_bouss            0.025059s - Marshall
   - [ ] MOM.F90                              0.010023s
      - [ ] extract_surface_state             0.010023s
   - [ ] MOM_interface_heights.F90            0.010023s
      - [ ] find_eta_2d                       0.005012s
      - [ ] find_eta_3d                       0.005012s
      - [ ] thickness_to_dz_3d                0.0s
   - [ ] MOM_coms.F90                         0.010023s
      - [ ] increment_ints_faster             0.010023s
   - [ ] mpp_comm_api.inc                     0.005012s
      - [ ] mpp_exit                          0.005012s
   - [ ] mpp_group_update.h                   0.005012s
      - [ ] mpp_do_group_update_r8            0.005012s
   - [ ] mpp_util_mpi.inc                     0.005012s
      - [ ] get_peset                         0.005012s
   - [ ] mpp_util.inc                         0.0s
      - [ ] lowercase                         0.0s
   - [ ] MOM_domain_infra.F90                 0.0s
      - [ ] create_vector_group_pass_2d       0.0s
   - [ ] MOM_forcing_type.F90                 0.0s
      - [ ] find_ustar_mech_forcing           0.0s
   - [ ] MOM_transcribe_grid.F90              0.0s
      - [ ] copy_dyngrid_to_mom_grid          0.0s
   - [ ] MOM_sum_output.F90                   0.0s
      - [ ] write_energy                      0.0s
   - [ ] MOM_diag_mediator.F90                0.0s
      - [ ] diag_copy_diag_to_storage         0.0s
      - [ ] diag_masks_set                    0.0s
      - [ ] diag_update_remap_grids           0.0s

These were collected with VTune:

.. code:: sh

   # run and collect stats
   mpiexec -n 1 vtune \
      -collect hotspots \
      -knob sampling-mode=hw \
      -knob enable-stack-collection=true \
      -r mom6-prof-vtune ../build/MOM6

   # generate report
   vtune \
      -report=hotspots \
      -r mom6-prof-vtune.gadi-login-??.gadi.nci.org.au \
      -format=csv \
      | column -ts $'\t' \
      | grep MOM6


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


Data Migration
--------------

We should aim minimize data transfer between the CPU host and GPU target.  This
is achieved by keeping the arrays on the GPU across multiple kernels.

Data directives are used to move an array between host and target.  This
operations occur outside of any compute kernels.

To move an array from host to device, or vice versa::

   !$omp target enter data map(to: x)

This allocates a new ``x`` on the GPU and sets the values from the CPU.  **It
will overwrite an existing x!**

To move data from GPU back to CPU::

   !$omp target exit data map(from: x)

**This will also deallocate x on the GPU.**

Arrays can be independently allocated or deleted on the GPU.  This block
allocates ``h`` on the GPU but does not fill its data.

.. code:: fortran

   allocate(CS%h(isd:ied,jsd:jed,nz))
   CS%h(:,:,:) = GV%Angstrom_H
   !$omp target enter data map(alloc: CS%h)

This block deallocates ``h`` on the GPU.

.. code:: fortran

   deallocate(CS%h)
   !$omp target exit data map(delete: h)

If you want to exchange values between a array which already exists on the GPU,
use ``update``.

.. code:: fortran

  !$omp target update to(h)
  call PressureForce(h, tv, CS%PFu, CS%PFv, G, GV, US, CS%PressureForce_CSp, &
                     CS%ALE_CSp, p_surf, CS%pbce, CS%eta_PF)
  !$omp target update from(CS%PFu, CS%PFv, CS%pbce, CS%eta_PF)

The ``to`` and ``from`` modifiers are with respect to the target GPU.

.. Where to add this?  (If at all?)

   OpenMP has a ``present()`` modifier to explicitly declare that an array is
   already on the target GPU.  But most compilers still do not support this
   modifier.  In Nvidia, the runtime appears to handle this well and avoids
   redundant transfers, so it is probably not necessary to use ``present()``.
   But this is still something that should be monitored closely.


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


Common Errors
-------------

Sadly, most errors are either generic

* (Runtime) "partially present"

  Typically this means that an array is not on the device.

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
