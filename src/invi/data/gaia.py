"""Load Gaia data from a csv file.

Note
----
1)  Gaia_source reference:
    https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html"""

import termcolor as _tc
import numpy as _np
from zero_point import zpt as _zpt

import fnc as _fnc
_pd = _fnc.utils.lazy.Import("pandas")

import invi as _invi

__all__ = ["load", "query"]

#-----------------------------------------------------------------------------

def _zero_point_corr(df):
    """Returns the parallax zero point correction and corrected parallax.

    Note
    ----
    1)  parallax_corr = parallax - correction
    2)  Units: [mas] (milliarcsecond, not micro)"""
    #--------------------------------------------------------------
    #Load data
    phot_g_mean_mag = df.phot_g_mean_mag.values
    nu_eff_used_in_astrometry = df.nu_eff_used_in_astrometry.values
    pseudocolour = df.pseudocolour.values
    ecl_lat = df.ecl_lat.values
    astrometric_params_solved = df.astrometric_params_solved.values #Valid values: 31, 95
    parallax = df.parallax.values #[mas]
    #--------------------------------------------------------------
    #The Zero Point Correction is provided for sources with 'pseudocolour'
    #in the range (1.24-1.72 mag):
    sel = (pseudocolour < 1.24) | (pseudocolour > 1.72)
    pseudocolour[sel] = _np.nan

    #The apparent magnitude 'phot_g_mean_mag' of one or more of the sources is outside the
    #expected range (6-21 mag).
    sel = (phot_g_mean_mag < 6.0) | (phot_g_mean_mag > 21.0)
    phot_g_mean_mag[sel] = _np.nan

    #The 'nu_eff_used_in_astrometry' of some of the 5p source(s) is outside
    #the expected range (1.1-1.9 mag).
    sel = (nu_eff_used_in_astrometry < 1.1) | (nu_eff_used_in_astrometry > 1.9)
    nu_eff_used_in_astrometry[sel] = _np.nan
    #--------------------------------------------------------------
    #Load tables
    _zpt.load_tables()

    #Zero point correction [mas]
    correction = _zpt.get_zpt(phot_g_mean_mag,
                              nu_eff_used_in_astrometry,
                              pseudocolour,
                              ecl_lat,
                              astrometric_params_solved)

    #Parallax corrected [mas]
    parallax_corrected = parallax - correction
    #--------------------------------------------------------------
    return correction, parallax_corrected


def _parallax(df):
    correction, parallax_corrected = _zero_point_corr(df)
    return {'parallax': df.parallax.values, #[mas]
            'zero_point_corr': correction, #[mas]
            'parallax_corrected': parallax_corrected #[mas]
            }

#-----------------------------------------------------------------------------

def _ICRS(df):
    return {'delta': df.dec.values, #[deg]
            'alpha': df.ra.values, #[deg]
            'mu_r': df.radial_velocity.values, #[km/s]
            'mu_delta': df.pmdec.values, #[mas/yr]
            'mu_alpha_str': df.pmra.values} #[mas/yr]

#-----------------------------------------------------------------------------

def _uncertainties(df):
    """Uncorrelated uncertainties."""
    parallax = df.parallax_error.values #[mas]

    cosd = _np.cos(_invi.units.deg_to_rad(df.dec.values))

    ICRS = {'delta': _invi.units.mas_to_deg(df.dec_error.values), #[deg]
            'alpha': _invi.units.mas_to_deg(df.ra_error.values/cosd), #[deg]
            'mu_r': df.radial_velocity_error.values, #[km/s]
            'mu_delta': df.pmdec_error.values, #[mas/yr]
            'mu_alpha_str': df.pmra_error.values} #[mas/yr]

    return {'parallax': parallax, 'ICRS': ICRS}

#-----------------------------------------------------------------------------

def _photometry_red(df):
    """Load apparent photometry.

    Note
    ----
    1)  The observed magnitudes and colour indices include reddening."""
    return {'g_red': df.phot_g_mean_mag.values, #[mag]
            'rp_red': df.phot_rp_mean_mag.values,
            'bp_red': df.phot_bp_mean_mag.values,
            'bprp_red': df.bp_rp.values}


def _photometry(df, cmd37_correction):
    """Apparent photometry with reddening correction.

    Note
    ----
    1)  The correction factor cmd37_correction is applied to improve the fitting with the CMD37 isochrone."""
    delta = df.dec.values #[deg]
    alpha = df.ra.values #[deg]
    g_red = df.phot_g_mean_mag.values #[mag]
    bprp_red = df.bp_rp.values #[mag]

    bprp, g = _invi.photometry.reddening.correction(delta, alpha, bprp_red, g_red, cmd37_correction)
    return {'bprp': bprp, 'g': g}

#-----------------------------------------------------------------------------

def _data_quality(df):
    """Source: https://arxiv.org/pdf/2402.01133.pdf

    RUWE < 1.4
    RUWE is the re-normalized unit weight error. Small values means well
    fitted by a single star model.

    ASTROMETRIC_EXCESS_NOISE_SIG < 2
    Excess noise refer to the extra noise in each observation which cause
    the residual scatter in the astrometric solution. And if
    ASTROMETRIC_EXCESS_NOISE_SIG is greater than two, then this excess
    noise can not be ignored.

    ASTROMETRIC_GOF_AL < 3
    This parameter represents the goodness of fit between the astrometric
    model and the observed data. A higher value indicates a poor fit.

    VISIBILITY_PERIODS_USED > 10
    This parameter represents the number of visibility periods used in
    the astrometric solution. A high number of visibility periods is a
    better indicator of an astrometrically well–observed source.

    Note
    ----
    1)  When including ASTROMETRIC_EXCESS_NOISE_SIG < 2 and ASTROMETRIC_GOF_AL < 3
    the brightest stars in the cluster are not selected. Perhaps it is better
    not to apply this cuts beforehand."""

    return {'ruwe': df.ruwe.values,
            'astrometric_excess_noise_sig': df.astrometric_excess_noise_sig.values,
            'astrometric_gof_al': df.astrometric_gof_al.values,
            'visibility_periods_used': df.visibility_periods_used.values,
            'duplicated_source': df.duplicated_source.values}

#-----------------------------------------------------------------------------

def _print_info(df, shuffle, random_state):
    #Number stars
    n_stars = df.shape[0]

    #Memory used by the data frame
    used_memory_B = df.memory_usage(deep=True).sum()
    used_memory = _fnc.utils.human_readable.memory(used_memory_B)

    _tc.cprint("Gaia:", 'light_blue')
    print(f"Number stars = {n_stars:_}")
    print(f" Used memory = {used_memory}")
    print(f"     Shuffle = {shuffle}")
    if shuffle:
        print(f"Random state = {random_state}")


def load(file_name, cmd37_correction, shuffle=False, random_state=None, verbose=True):
    """Load Gaia data from a csv file."""
    #-------------------------------------------
    #Load data frame
    df = _pd.read_csv(file_name).fillna(_np.nan)
    #Shuffle rows
    if shuffle:
        df = df.sample(frac=1, replace=False, random_state=random_state).reset_index(drop=True)
    #-------------------------------------------
    #Definition dictionary
    stars = {}
    stars['source_id'] = df.source_id.values
    stars['random_index'] = df.random_index.values
    stars['parallax'] = _parallax(df)
    stars['ICRS'] = _ICRS(df)
    stars['uncertainties'] = _uncertainties(df)
    stars['photometry_red'] = _photometry_red(df)
    stars['photometry'] = _photometry(df, cmd37_correction)
    stars['data_quality'] = _data_quality(df)
    #-------------------------------------------
    if verbose:
        _print_info(df, shuffle, random_state)
    #-------------------------------------------
    return stars

#-----------------------------------------------------------------------------

def query(dec_cut, points_polygon, verbose=True):
    #----------------------------
    polygon = "1 = CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', "
    for i in range(len(points_polygon)-1):
        polygon += f"{points_polygon[i][0]},{points_polygon[i][1]}, "
    polygon += f"{points_polygon[-1][0]},{points_polygon[-1][1]}))"
    #----------------------------
    adql = f"""SELECT *
FROM gaiadr3.gaia_source
WHERE {polygon}
AND dec BETWEEN {dec_cut[0]} AND {dec_cut[1]}
AND parallax < 1.0/0.3
AND bp_rp IS NOT NULL
AND b > 15.0
AND ruwe < 1.2
AND visibility_periods_used >= 10
AND duplicated_source = False;"""
    #----------------------------
    if verbose:
        print(adql)
    #----------------------------
    return adql

#-----------------------------------------------------------------------------
