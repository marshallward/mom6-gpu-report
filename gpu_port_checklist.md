
# MOM6 `double_gyre` GPU Porting Progress

Below are a list of subroutines/functions that are used in the `double_gyre`
test - sorted first by source files which use up the most CPU time, then by the
subroutines/functions in those sources files which use up the most time.

- [ ] MOM_continuity_PPM.F90               1.207831s
   - [x] meridional_flux_adjust            0.205482s
   - [ ] zonal_flux_adjust                 0.170399s **Edward**
   - [ ] zonal_flux_layer                  0.095223s **Edward**
   - [ ] set_merid_bt_cont                 0.080188s
   - [x] merid_flux_layer                  0.070164s
   - [ ] set_zonal_bt_cont                 0.060141s
   - [ ] ppm_reconstruction_x              0.045106s
   - [ ] zonal_mass_flux                   0.045106s
   - [ ] ppm_reconstruction_y              0.040094s
   - [ ] zonal_flux_thickness              0.040094s
   - [ ] meridional_mass_flux              0.030070s
   - [ ] ppm_limit_pos                     0.030070s
   - [ ] meridional_flux_thickness         0.020047s
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
- [ ] MOM_CoriolisAdv.F90                  0.125294s **Marshall**
   - [ ] coradcalc                         0.090211s **Marshall**
   - [ ] gradke                            0.035082s **Marshall**
- [ ] diag_manager.F90                     0.070164s
   - [ ] send_data_3d                      0.070164s
- [ ] MOM_set_viscosity.F90                0.055129s
   - [ ] set_viscous_bbl                   0.055129s
- [ ] MOM_dynamics_split_RL2.F90           0.035082s
   - [ ] step_mom_dyn_split_rk2            0.030070s
   - [ ] register_restarts_dyn_split_rk2   0.005012s
- [ ] MOM_PressureForce_FV.F90             0.025059s **Marshall**
   - [ ] pressureforce_fv_bouss            0.025059s **Marshall**
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
