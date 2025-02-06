
# MOM6 `double_gyre` GPU Porting Progress

Below are a list of subroutines/functions that are used in the `double_gyre`
test - sorted first by source files which use up the most CPU time, then by the
subroutines/functions in those sources files which use up the most time.

- [ ] MOM_continuity_PPM.F90               1.207831s
   - [ ] meridional_flux_adjust            0.205482s **Edward**
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
