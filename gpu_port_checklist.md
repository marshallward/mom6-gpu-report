
# MOM6 `double_gyre` GPU Porting Progress

Below are a list of subroutines/functions that are used in the `double_gyre`
test - sorted first by source files which use up the most CPU time, then by the
subroutines/functions in those sources files which use up the most time.

- [x] MOM_continuity_PPM.F90               1.207831s **Edward** [**first draft**](https://github.com/edoyango/MOM6/tree/fuse-loops-gpu-port)
   - [x] meridional_flux_adjust            0.205482s **Edward**
   - [x] zonal_flux_adjust                 0.170399s **Edward**
   - [x] zonal_flux_layer                  0.095223s **Edward**
   - [x] set_merid_bt_cont                 0.080188s **Edward**
   - [x] merid_flux_layer                  0.070164s **Edward**
   - [x] set_zonal_bt_cont                 0.060141s **Edward**
   - [x] ppm_reconstruction_x              0.045106s **Edward**
   - [x] zonal_mass_flux                   0.045106s **Edward**
   - [x] ppm_reconstruction_y              0.040094s **Edward**
   - [x] zonal_flux_thickness              0.040094s **Edward**
   - [x] meridional_mass_flux              0.030070s **Edward**
   - [x] ppm_limit_pos                     0.030070s **Edward**
   - [x] meridional_flux_thickness         0.020047s **Edward**
   - [x] meridional_edge_thickness         0.005012s **Edward**
   - [x] continuity_merdional_convergence  0.0s      **Edward**
   - [x] continuity_ppm                    0.0s      **Edward**
   - [x] continuity_zonal_convergence      0.0s      **Edward**
   - [x] zonal_edge_thickness              0.0s      **Edward**
- [ ] MOM_barotropic.F90                   1.182772s
   - [ ] btstep                            1.002349s
   - [ ] set_local_bt_cont_types           0.080188s
   - [ ] find_uhbt                         0.040094s
   - [ ] find_vhbt                         0.040094s
   - [ ] bt_mass_source                    0.010023s
   - [ ] btcalc                            0.010023s
- [x] MOM_vert_friction.F90                0.726703s **Edward** [**draft**](https://github.com/edoyango/MOM6/tree/vertvisc-gpu)
   - [x] vertvisc_coef                     0.355834s **Edward**
   - [x] vertvisc                          0.140329s **Edward**
   - [x] vertvisc_remnant                  0.120282s **Edward**
   - [x] find_coupling_coef                0.075176s **Edward**
   - [x] vertvisc_limit_vel                0.035082s **Edward**
- [x] MOM_hor_visc.F90                     0.200470s **Marshall**
   - [x] horizontal_viscosity              0.200470s **Marshall**
- [x] MOM_CoriolisAdv.F90                  0.125294s **Marshall**
   - [x] coradcalc                         0.090211s **Marshall**
   - [x] gradke                            0.035082s **Marshall**
- [ ] diag_manager.F90                     0.070164s
   - [ ] send_data_3d                      0.070164s
- [ ] MOM_set_viscosity.F90                0.055129s
   - [ ] set_viscous_bbl                   0.055129s
- [ ] MOM_dynamics_split_RK2.F90           0.035082s **Marshall**
   - [ ] step_mom_dyn_split_rk2            0.030070s **Marshall**
   - [ ] register_restarts_dyn_split_rk2   0.005012s **Marshall**
- [x] MOM_PressureForce_FV.F90             0.025059s **Marshall**
   - [x] pressureforce_fv_bouss            0.025059s **Marshall**
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

```bash
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
```

## `MOM_continuity_PPM.F90`

### Loop refactoring

It was discovered that porting loops as they were was causing performance degredation.
This was because long 3D loops were written such that the outer-most loop was initiated
in one subroutine, and the inner loops were called in another subroutine, perhaps one
or two subroutines deep. Adding OpenMP target directives on the inner-most loop caused
significant slow down due to latency in kernel launches. A prime example of this
problem is [`merid_flux_layer`](https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/core/MOM_continuity_PPM.F90#L1787), 
which is called by [`meridional_flux_adjust`](https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/core/MOM_continuity_PPM.F90#L1995) in a
k-loop, which is in-turn called in a newton iterative loop. This subroutine is then
called in a `j` outer loop in [`meridional_mass_flux](https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/core/MOM_continuity_PPM.F90#L1413).
So, if the loops in `merid_flux_layer` are ported, that GPU kernel would be 
called `nk*nj*niter*ntimesteps` times (where `nk/j` is number of `k/j` iterations, and 
`niter` is an average number of newton iteratios). In `double_gyre`, this amounts to 
roughly `2*40*3*1240 ~ 3e6` calls, where each call takes ~40μs (~30μs compute + 10μs launch
latency), which is ~12s in total (the entire `double_gyre` CPU-only case takes about 10s). 
Note that `meridional_flux_adjust` is called in [another place](https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/core/MOM_continuity_PPM.F90#L2223C3-L2225C60), 
and `meridional_flux_adjust` is called in a [few](https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/core/MOM_continuity_PPM.F90#L1768)
[other](https://github.com/NOAA-GFDL/MOM6/blob/e818ea4e792f0b85797247f955789b3c1210db8d/src/core/MOM_continuity_PPM.F90#L2265) 
places (separate from `meridional_flux_adjust`). Furthermore, `zonal_flux_layer`
follows a similar pattern and does a similar amount of work.

Refactoring the loops such that they occur in blocks of 3D loops improved mitigated
this significantly as number of kernel launches can be made `O(ntimesteps*niter)`. The
kernels themselves also have more parallelism for the GPU to exploit. 

However, writing loops in this way mean that the iterative loop in 
`zonal/meridional_flux_adjust` has to occur on the entire grid together, rather than
on a per-row basis like in the original code. This means more work is being done.
In `double_gyre`, this extra work is roughly 30% as the original code stops
iterating on rows where all elements have converged.

This all results in a significant speedup on the GPU, but a meaningful slowdown on the
CPU (1.3-1.5x slower).

### `meridional_flux_adjust`

* a couple raw assignments had to be turned into loops to be ported
* any uses of a derived type member requires the entire instance to be sent to 
the GPU (I'm not sure why yet).
* useful to send only the part of the arrays that's needed.
* The newton-raphson loop has dependencies, so naturally the the inner loops are
ported and not the iterative loop.
* using reduction clauses inside a loop seems to work ok, even if the loop is
doing other things.
* `merid_flux_layer` is called in a loop (iterates over `k` outside of
`merid_flux_layer`, iterates over `i` inside). *I think* for best performance, 
these loops are fused into one kernel, but that may be challenging to refactor. 
For now, the outer `k` loop is left as-is, and the `i` loop in 
`merid_flux_layer` is ported.
* After porting just the loops of this subroutine and `merid_flux_layer`, the
`double_gyre` case takes 265s (up from ~20s). 
* In `target enter/exit data` statements, if slices are mapped `to`, then the
same slice must also be mapped `from/release`. `from/release` the whole arrays
doesn't work (will get a runtime error:  
`variable in data clause is partially present on the device`)
* When writing `target enter/exit data` directives, remember to wrap `optional`
variables in an if statement.
* After `enter/exit` statements are written to keep data on GPU, time taken
is down to 182.6s (down 83s), but still much slower as `meridional_flux_adjust`
is still called ~120k times leading to lots of data transfer to/from GPU.
* After fusing some of the kernels, brought down the time to 171.5s
   * Conditionals that determine whether loops are executed didn't seem to like
   being inside `target` regions (would slow down the execution).

#### profiling

gotta use openacc to get useful data transfer info e.g. `nsys profile -t openacc --stats=true -b dwarf --force-overwrite true -o MOM6-test ../build/MOM6`.

```
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)   StdDev (ns)                      Name                    
 --------  ---------------  ----------  -----------  -----------  ---------  ----------  -----------  --------------------------------------------
     19.8   88,912,105,446     982,490     90,496.7    117,935.0      5,737  28,173,420    500,124.7  Enter Data@MOM_continuity_PPM.F90:1831      
     16.1   72,172,684,260     982,490     73,459.0     67,433.0      9,460  22,459,340    389,026.8  Exit Data@MOM_continuity_PPM.F90:1831       
     14.9   66,969,698,849  10,628,840      6,300.8      4,970.0      2,803  22,192,119    139,040.5  Enqueue Upload@MOM_continuity_PPM.F90:1831  
      7.9   35,240,696,361     118,080    298,447.6    224,810.5    205,968  22,468,954    955,788.6  Enter Data@MOM_continuity_PPM.F90:2073      
      5.4   24,092,528,059   3,896,640      6,182.9      4,378.0      3,517  22,119,586    143,681.0  Enqueue Upload@MOM_continuity_PPM.F90:2073  
      4.7   20,856,196,938   3,188,652      6,540.8      5,140.0      3,645  22,318,711    141,954.5  Enqueue Upload@MOM_continuity_PPM.F90:1863  
      4.3   19,474,651,883   1,062,884     18,322.5     12,167.0     10,402  22,146,178    168,007.3  Enqueue Download@MOM_continuity_PPM.F90:1863
      4.3   19,065,764,765     982,490     19,405.6      3,551.0      1,852  22,113,530    175,299.3  Wait@MOM_continuity_PPM.F90:1863            
      2.3   10,293,150,013     687,208     14,978.2     17,392.0      4,630  21,917,725    180,992.5  Exit Data@MOM_continuity_PPM.F90:2131       
      2.1    9,361,336,007     118,080     79,279.6     60,397.0     56,211  22,174,930    453,956.5  Exit Data@MOM_continuity_PPM.F90:2217       
      1.8    8,054,287,557     687,208     11,720.3     12,267.0      4,717  22,104,992    165,771.7  Enter Data@MOM_continuity_PPM.F90:2131      
```

Lets of enter/exit data statements and synchronizes in the nsys gui.

### `merid_flux_layer`

* The first loop is standard, but for some reason `vh(ish:ieh)` needs to be 
mapped as `tofrom` even though the entire slice appears to be overwritten.
* I'm avoiding adding `target enter data` statements in this subroutine 
as the program got slower when I added them (for this subroutine). I'm also not
sure what happens when multiple `target enter data` address the same data.
* **Have yet to test whether the second loop (guarded by `if (local_open_BC) then`) was ported correctly**

### zonal_flux_adjust

* close to identical structure to `meridional_flux_adjust`

### zonal_flux_layer

* close to identical structure to `merid_flux_layer`
* **Have yet to test whether the second loop (guarded by `if (local_open_BC) then`) was ported correctly**

### set_merid_bt_cont

* This calls `meridional_flux_adjust` and `merid_flux_layer`
   * Input variables that are `intent(out)` in the child subroutine
   needed to be `alloc`/`to` entirely, not just a section
   * Had trouble with nested `enter target data` as subsequent entries were 
   overwriting data on the device with outdated data on the host.
* For some reason, I needed a `target update from(dvR, dvL)` statement in an
innocuous spot to get correct results.
   * This turned out to be because of missing explicit mappings for a couple
   variables, so the compiler was generating implicit `tofrom` for them. Adding
   the correct explicit mappings resolved it.

### set_zonal_BT_cont

* This call `zonal_flux_adjust` and `zonal_flux_layer`
* Like `set_merid_bt_cont`, I had trouble with the `enter/exit/update`
statements. 
   * For some reason, `u` needed to be released before the final loop, but if I
   move the release of `u` to the end with the rest of the releases, I get the
   wrong results.
      * This stopped being an issue after enter/exit data statements were added
      in `zonal_mass_flux`, which calls this subroutine.

### zonal_mass_flux

* There's quite a few if statements guarding loops. I've left them un-ported as
`double_gyre` doesn't use them. Tagged the sections with `not in double_gyre`
comment.
* `BT_cont` is a pointer, but including it in the final `exit data map(from:)`
statement caused an error. Leaving it out and letting the compiler generate an
implicit one seemed to work. The member arrays needed are explicitly specified.
   * adding `BT_cont` as an `alloc` in the enter statement and `release` in the
   exit statement prevented re-transerring `BT_cont`. I'm guessing `to` doesn't
   work because it transers the pointer address, which points to the wrong place
   on the GPU?
* This calls `zonal_flux_layer` and `set_zonal_BT_cont` quite a few times as the
main loop iterates over `j` and calls those subroutines within the loop. 
Mapping entire arrays with `enter/exit` reduced data transfers significantly.
* I'm not sure what the logic is behind the compiler's decision, but I noticed
that slicing multiple indices for rank 3+ arrays when transferring data will 
result in many transfers being initiated. For rank3+ arrays, it seems like only
the first index can be sliced in data transfer statements to reduce the number
of transfers initiated. 

### meridional_mass_flux

Very similar to `zonal_mass_flux`.

### zonal_flux_thickness and meridional_flux_thickness

* These two consisted of two loops that were straightforward to port. There are
loops in each that weren't ported as they weren't being used in `double_gyre`.
* `enter/exit data` statements haven't been added as they were only being called
by `zonal_mass_flux` and `meridional_mass_flux`, respectively and the
`enter/exit data` statements in those subroutines mitigated the need for any
transfers.

## MOM_continuity_PPM.F90

### `vertvisc_coef`  and `find_coupling_coef`

`vertvisc_coef` uses a similar `jki` loop format like in `MOM_continuity_PPM.F90`.
Inside the big `jki` loops, `find_coupling_coef` is called for each `j` iteration.
Since in the continuity solver, separating the big `jki` loop into smaller `kji`
loops resulted in poor CPU performance, I started off trying to use nested
parallelism to port the `jki` loops in `vertvisc_coef`. I gave up on that pretty
quickly because I kept getting CUDA memory errors, and couldn't find the source.

I ended up porting these two subroutines by seperating out the loops as it was
much easier to progressively port and check correctness. These loops were ported
with `do concurrent`, which worked seamlessly and with good performance when
coupled with appropriate data mapping clauses.

### `vertvisc`

Again, uses two `jki` loops - one perfomed on the u-points, and the other for the
`v-points`. However, they were simpler than in `continuity_ppm` and `vertvisc-coef`
as they didn't call any subroutines. This made it easier to try to port the entire
`jki` using nested parallelism.

In the process, I discovered a bug with NVHPC 25.3 when an allocatable array's
allocation status is checked within an OpenMP ported loop ([my post on nvidia dev
forums](https://forums.developer.nvidia.com/t/bug-nvhpc-25-3-and-checking-unallocated-fortran-arrays-in-openmp-target-loops/333128)).

I ported `vertvisc` with both the nested parallelism strategy ([branch]
(https://github.com/edoyango/MOM6/blob/e1db176d10b8fc0c464ba7fc52a9b9c77ca10e35/src/parameterizations/vertical/MOM_vert_friction.F90#L542)), 
and the separated loop strategy ([branch](https://github.com/edoyango/MOM6/blob/cac9d83959bea5cbb30290f7a146668b685e53a8/src/parameterizations/vertical/MOM_vert_friction.F90#L542), 
and found the latter to result in better performance (when tested using the 
`benchmark` case).

### `vertvisc_remnant`

Like `vertvisc`, this consisted of simpler `jki` loops which I could port wholly
using nested parallelism.
