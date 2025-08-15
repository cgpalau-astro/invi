"""Load data from a csv file and ADQL query for the globular cluster cone
selection."""

import termcolor as _tc
import numpy as _np

import fnc as _fnc
_cl = _fnc.utils.lazy.Import("sklearn.cluster")

import invi as _invi

__all__ = ["load", "query", "cuts"]

#-----------------------------------------------------------------------------

def _components(gc, eps):
    #-----------------------------
    def normalitsation_data(x, y):
        x = x - _np.mean(x)
        y = y - _np.mean(y)
        x = x / _np.std(x)
        y = y / _np.std(y)
        return _np.array([x, y]).T

    def clustering_dbscan(x, y, eps, min_samples=5):
        #Normalise data
        data = normalitsation_data(x, y)
        #DBSCAN clustering
        dbscan = _cl.DBSCAN(eps=eps, min_samples=min_samples, algorithm="brute").fit(data)
        lab = dbscan.labels_
        #Select outliers
        outliers = lab == -1
        return outliers
    #-----------------------------
    g_red = gc['photometry_red']['g_red']
    bprp_red = gc['photometry_red']['bprp_red']
    outliers = clustering_dbscan(g_red, bprp_red, eps)
    return {'outliers': outliers,
            'gc': _np.logical_not(outliers)}

#-----------------------------------------------------------------------------

def load(file_name, prm_gc, shuffle=False, random_state=None, verbose=True):
    """Load Gaia-DR3 data for globular cluster file."""

    #Load gaia data
    gc = _invi.data.gaia.load(file_name, prm_gc['mock']['cmd37_correction'], shuffle, random_state, verbose)

    gc['components'] = _components(gc, prm_gc['gaia']['outliers_eps'])

    #Heliocentric distance globular cluster
    distance = prm_gc['ICRS']['r']

    #Absolut photometry
    gc['photometry']['BPRP'] = gc['photometry']['bprp']
    gc['photometry']['G'] = _invi.photometry.magnitudes.m_to_M(gc['photometry']['g'], distance)

    return gc

#-----------------------------------------------------------------------------

def query(cuts, prm_gc, verbose=True):
    """ADQL query globular clusters Gaia-DR3.

    Web: https://gaia.aip.de/query/

    Example
    -------
    cuts = {'parallax_i': -2.0, #[mas]
            'parallax_s': 2.0, #[mas]
            #-----------------------------
            'angle_aperture': 0.15, #0.25 #[deg]
            #-----------------------------
            'mod_pm': 1.0, #2.0 #[mas/yr]
            #-----------------------------
            'bp_rp_i': -0.1, #[mag]
            'bp_rp_s': 1.6, #[mag]
            #-----------------------------
            'ruwe': 1.2, #1.4
            'visibility_periods_used': 10,
            'duplicated_source': False}"""
    #--------------------------------------------
    def isgn(x):
        """Returns inverted sgn as character."""
        if _np.sign(x) > 0.0:
            return "-"
        return "+"
    #--------------------------------------------
    ra = prm_gc['ICRS']['alpha']
    dec = prm_gc['ICRS']['delta']
    a = prm_gc['ICRS']['mu_alpha_str']
    d = prm_gc['ICRS']['mu_delta']
    #--------------------------------------------
    adql = f"""SELECT *
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {cuts['angle_aperture']}) )
AND parallax BETWEEN {cuts['parallax_i']} AND {cuts['parallax_s']}
AND SQRT((pmra{isgn(a)}{abs(a)})*(pmra{isgn(a)}{abs(a)}) + (pmdec{isgn(d)}{abs(d)})*(pmdec{isgn(d)}{abs(d)})) <= {cuts['mod_pm']}
AND bp_rp BETWEEN {cuts['bp_rp_i']} AND {cuts['bp_rp_s']}
AND ruwe < {cuts['ruwe']}
AND visibility_periods_used >= {cuts['visibility_periods_used']}
AND duplicated_source = {cuts['duplicated_source']};"""
    #--------------------------------------------
    if verbose:
        _tc.cprint("Information:", "light_blue")
        print(f"Cut distance: {1.0/cuts['parallax_i']} - {1.0/cuts['parallax_s']} [kpc]\n")
        _tc.cprint("ADQL query:", "light_blue")
        print(adql)
    #--------------------------------------------
    return adql

#-----------------------------------------------------------------------------

def cuts(prm_gc, sample_icrs_dict, sample_parallax_dict, sample_phot_red):
    """Cuts mocking the star selection in the globular cluster query.

    Note
    ----
    1)  The 'bprp_cut' is not applied because it eliminates the stars with the
        largest uncertainties. This is not observed in the real data, perhaps
        because the limits of the simulation of the errors of bp and rp magnitudes
        are not accurately set."""

    parallax = sample_parallax_dict['parallax']

    ra = prm_gc['ICRS']['alpha']
    dec = prm_gc['ICRS']['delta']
    mu_alpha_str = prm_gc['ICRS']['mu_alpha_str']
    mu_delta = prm_gc['ICRS']['mu_delta']

    bprp = sample_phot_red['bprp_red']

    d = _invi.misc.euclidean_distance(sample_icrs_dict['delta'], sample_icrs_dict['alpha'], dec, ra) #[deg]
    d_mu = _invi.misc.euclidean_distance(sample_icrs_dict['mu_delta'], sample_icrs_dict['mu_alpha_str'], mu_delta, mu_alpha_str) #[mas/yr]

    plx_cut = (parallax > -2.0) & (parallax < 2.0)

    bprp_cut = (bprp > -0.1) & (bprp < 1.6)

    return (d <= 0.15) & (d_mu <= 1.0) & plx_cut #& bprp_cut

#-----------------------------------------------------------------------------
