//----------------------------------------------------------------------------

void intersections(int n_lines_file_data_extract, char *file_data_extract, char *file_intersec, char *file_orbit_bundle) {

    //Definition variables
    char line[1024];
    double intersec;

    struct WS ws;
    ws.w = calloc(D, sizeof(double));
    ws.S = calloc(D*D, sizeof(double));

    //Open file data stars and skip first line
    FILE *f_data_extract = fopen(file_data_extract, "r");
    fgets(line, sizeof(line), f_data_extract);

    //Create file to store intersections and write header
    FILE *f_intersec = fopen(file_intersec, "w");
    fprintf(f_intersec, "intersec\n");

    //Load volume defined by orbit bundle
    int n_ob = count_file_lines(file_orbit_bundle);
    struct WS *volume_ob = load_volume_ob(file_orbit_bundle, n_ob);

    //Compute intersections
    for (int i = 0; i < n_lines_file_data_extract-1; i++) {
        //Get star
        fgets(line, sizeof(line), f_data_extract);
        ws = split_line_data(line, ws, 1);

        //Compute convolution
        intersec = gaussian_convolution(ws, volume_ob, n_ob);

        //Print intersection to file
        fprintf(f_intersec, "%+.15E\n", intersec);
    }

    //Close files and free memory
    fclose(f_data_extract);
    fclose(f_intersec);
    free_WS(volume_ob, n_ob);
    free(ws.w);
    free(ws.S);
}

//----------------------------------------------------------------------------

void pre_selection(double eps, int n_lines_file_data, char *file_data, char *file_intersec, char *file_pre_sel, char *file_intersec_pre_sel) {

    //Definition variables
    char line[16384];
    double intersec;

    //Number pre-selected stars
    //int n_pre = 0;

    //Open file data stars and skip first line
    FILE *f_data = fopen(file_data, "r");
    fgets(line, sizeof(line), f_data);

    //Open file intersections and skip first line
    FILE *f_intersec = fopen(file_intersec, "r");
    fgets(line, sizeof(line), f_intersec);

    //Create file pre-selected stars and write header
    FILE *f_pre_sel = fopen(file_pre_sel, "w");
    fprintf(f_pre_sel, "solution_id,designation,source_id,random_index,ref_epoch,ra,ra_error,dec,dec_error,parallax,parallax_error,parallax_over_error,pm,pmra,pmra_error,pmdec,pmdec_error,ra_dec_corr,ra_parallax_corr,ra_pmra_corr,ra_pmdec_corr,dec_parallax_corr,dec_pmra_corr,dec_pmdec_corr,parallax_pmra_corr,parallax_pmdec_corr,pmra_pmdec_corr,astrometric_n_obs_al,astrometric_n_obs_ac,astrometric_n_good_obs_al,astrometric_n_bad_obs_al,astrometric_gof_al,astrometric_chi2_al,astrometric_excess_noise,astrometric_excess_noise_sig,astrometric_params_solved,astrometric_primary_flag,nu_eff_used_in_astrometry,pseudocolour,pseudocolour_error,ra_pseudocolour_corr,dec_pseudocolour_corr,parallax_pseudocolour_corr,pmra_pseudocolour_corr,pmdec_pseudocolour_corr,astrometric_matched_transits,visibility_periods_used,astrometric_sigma5d_max,matched_transits,new_matched_transits,matched_transits_removed,ipd_gof_harmonic_amplitude,ipd_gof_harmonic_phase,ipd_frac_multi_peak,ipd_frac_odd_win,ruwe,scan_direction_strength_k1,scan_direction_strength_k2,scan_direction_strength_k3,scan_direction_strength_k4,scan_direction_mean_k1,scan_direction_mean_k2,scan_direction_mean_k3,scan_direction_mean_k4,duplicated_source,phot_g_n_obs,phot_g_mean_flux,phot_g_mean_flux_error,phot_g_mean_flux_over_error,phot_g_mean_mag,phot_bp_n_obs,phot_bp_mean_flux,phot_bp_mean_flux_error,phot_bp_mean_flux_over_error,phot_bp_mean_mag,phot_rp_n_obs,phot_rp_mean_flux,phot_rp_mean_flux_error,phot_rp_mean_flux_over_error,phot_rp_mean_mag,phot_bp_rp_excess_factor,phot_bp_n_contaminated_transits,phot_bp_n_blended_transits,phot_rp_n_contaminated_transits,phot_rp_n_blended_transits,phot_proc_mode,bp_rp,bp_g,g_rp,radial_velocity,radial_velocity_error,rv_method_used,rv_nb_transits,rv_nb_deblended_transits,rv_visibility_periods_used,rv_expected_sig_to_noise,rv_renormalised_gof,rv_chisq_pvalue,rv_time_duration,rv_amplitude_robust,rv_template_teff,rv_template_logg,rv_template_fe_h,rv_atm_param_origin,vbroad,vbroad_error,vbroad_nb_transits,grvs_mag,grvs_mag_error,grvs_mag_nb_transits,rvs_spec_sig_to_noise,phot_variable_flag,l,b,ecl_lon,ecl_lat,in_qso_candidates,in_galaxy_candidates,non_single_star,has_xp_continuous,has_xp_sampled,has_rvs,has_epoch_photometry,has_epoch_rv,has_mcmc_gspphot,has_mcmc_msc,in_andromeda_survey,classprob_dsc_combmod_quasar,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_star,teff_gspphot,teff_gspphot_lower,teff_gspphot_upper,logg_gspphot,logg_gspphot_lower,logg_gspphot_upper,mh_gspphot,mh_gspphot_lower,mh_gspphot_upper,distance_gspphot,distance_gspphot_lower,distance_gspphot_upper,azero_gspphot,azero_gspphot_lower,azero_gspphot_upper,ag_gspphot,ag_gspphot_lower,ag_gspphot_upper,ebpminrp_gspphot,ebpminrp_gspphot_lower,ebpminrp_gspphot_upper,libname_gspphot,pos\n");

    //Create file pre-selected intersections and write header
    FILE *f_intersec_pre_sel = fopen(file_intersec_pre_sel, "w");
    fprintf(f_intersec_pre_sel, "intersec\n");

    //Determine intersections above eps
    for (int i = 0; i < n_lines_file_data-1; i++) {
        //Get intersection
        fgets(line, sizeof(line), f_intersec);
        sscanf(line, "%lf", &intersec);

        //Get star data
        fgets(line, sizeof(line), f_data);

        if (intersec >= eps) {
            fprintf(f_pre_sel, "%s", line);
            fprintf(f_intersec_pre_sel, "%+0.15E\n", intersec);
            //n_pre += 1;
        }
    }

    //Print results
    //printf("eps = %+0.1E\n", eps);
    //printf("Number pre-selected = %d\n", n_pre);
    //printf("\n");

    //Close files
    fclose(f_data);
    fclose(f_intersec);
    fclose(f_pre_sel);
    fclose(f_intersec_pre_sel);
}

//----------------------------------------------------------------------------
