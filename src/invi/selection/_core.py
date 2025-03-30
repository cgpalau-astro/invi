"""Function for the pre-selection and selection of stars.

Note
----
1)  Create the shared object: gcc -Wall -O3 -lm -shared -o master.so -fPIC master.c"""

import os as _os
import ctypes as _ctypes
from pathlib import Path as _Path

import scipy as _scipy
import numpy as _np
import tqdm as _tqdm

import fnc as _fnc
import invi as _invi

_pd = _fnc.utils.lazy.Import("pandas")

__all__ = ["random_parameters", "bck_frw", "spline", "write_to_file",
           "extract",
           "intersections", "pre_selection", "count_file_lines",
           "with_neighbours", "cuts", "final_selection"]

#-----------------------------------------------------------------------------

def random_parameters(random_state, prm_gc, prm_mw, scale):
    """Random initial conditions for a cluster and random parameters for the halo."""
    _np.random.seed(random_state)

    #Globular cluster ICRS position
    icrs = prm_gc['ICRS']

    #Dark halo
    halo = prm_mw['halo']

    ICRS = {'r': _np.random.normal(icrs['r'], scale[0], 1)[0],
            'delta': _np.random.normal(icrs['delta'], scale[1], 1)[0],
            'alpha': _np.random.normal(icrs['alpha'], scale[2], 1)[0],
            'mu_r': _np.random.normal(icrs['mu_r'], scale[3], 1)[0],
            'mu_delta': _np.random.normal(icrs['mu_delta'], scale[4], 1)[0],
            'mu_alpha_str': _np.random.normal(icrs['mu_alpha_str'], scale[5], 1)[0]}

    halo = {'rho': _np.random.normal(halo['rho'], scale[6], 1)[0],
            'a': _np.random.normal(halo['a'], scale[7], 1)[0],
            'q': _np.random.normal(halo['q'], scale[8], 1)[0],
            'beta': _np.random.normal(halo['beta'], scale[9], 1)[0]}

    return ICRS, halo


def bck_frw(gc, prm, T, N):
    """Compute an orbit backwards and forwards given the initial condition of
    a star. The data returned corresponds to the orbit forwards."""

    #Milky Way potential (galpy)
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Initial condition bck
    ic_icrs_bck = _invi.dicts.dict_to_array(gc['ICRS'])
    ic_fsr_bck = _invi.coordinates.ICRS_to_FSR(ic_icrs_bck, prm['sun'], mw.v_lsr)

    #Inverse orbit bck
    _t_bck, orb_fsr_bck = _invi.galpy.orbit.integrate(ic_fsr_bck, mw.potential, -T, N)

    #Initial condition frw
    ic_fsr_frw = orb_fsr_bck.T[-1]

    #Forward orbit
    _t_frw, orb_fsr_frw = _invi.galpy.orbit.integrate(ic_fsr_frw, mw.potential, T*2, N*2)
    orb_icrs_frw = _invi.coordinates.FSR_to_ICRS(orb_fsr_frw, prm['sun'], mw.v_lsr)

    #Dictionary including the orbit frw
    orbit = _invi.dicts.array_to_dict(orb_icrs_frw, 'ICRS')

    return orbit


def spline(x, n_points_spline):
    """Re-parameterise the orbit with a defined number of points."""

    #Definition spline
    l = _np.linspace(0.0, 1.0, len(x))
    spl = _scipy.interpolate.splrep(l, x, s=0)

    #Evaluation spline
    p = _np.linspace(0.0, 1.0, n_points_spline)
    x_new = _scipy.interpolate.splev(p, spl, der=0)

    return x_new


def write_to_file(fw, w, S, num_frmt='+0.15E'):

    def flatten_symmetric(S):
        """Flatten the diagonal and the superior triangle of a matrix."""
        I, J = _np.shape(S)
        flat = []
        for i in range(0, I):
            for j in range(i, J):
                flat.append(S[i][j])
        return flat

    line = ' '.join([' '.join([f"{x:{num_frmt}}" for x in w]),
                     ' '.join([f"{x:{num_frmt}}" for x in flatten_symmetric(S)]),
                     '\n'])

    fw.write(line)

#-----------------------------------------------------------------------------

def extract(file_input, file_output):
    """Extract phase-space, uncertainties, and g, bprp magnitudes from a full gaia catalogue."""
    df = _pd.read_csv(file_input).fillna(_np.nan)

    #pmra = pmra_str,
    #ra_error = ra_error_str
    #pmra_error = pmra_error_str

    #Columns to be extracted
    keys = ['parallax', 'dec', 'ra', 'radial_velocity', 'pmdec', 'pmra', 'parallax_error', 'dec_parallax_corr', 'ra_parallax_corr', 'parallax_pmdec_corr', 'parallax_pmra_corr', 'dec_error', 'ra_dec_corr', 'dec_pmdec_corr', 'dec_pmra_corr', 'ra_error', 'ra_pmdec_corr', 'ra_pmra_corr', 'radial_velocity_error', 'pmdec_error', 'pmra_pmdec_corr', 'pmra_error', 'bp_rp', 'phot_g_mean_mag']

    data = {}
    for key in keys:
        data[key] = df[key].values

    #Extract and save as csv
    _pd.DataFrame(data).to_csv(file_output, index=False, na_rep='NaN')

#-----------------------------------------------------------------------------

#Absolute path to master.so:
_base_path = _Path(__file__).parent
_file_path = (_base_path / "master.so").resolve()

_clib = _ctypes.CDLL(_file_path)
_clib.intersections.argtypes = [_ctypes.c_int64, _ctypes.c_char_p, _ctypes.c_char_p, _ctypes.c_char_p]
_clib.pre_selection.argtypes = [_ctypes.c_double, _ctypes.c_int64, _ctypes.c_char_p, _ctypes.c_char_p, _ctypes.c_char_p, _ctypes.c_char_p]


@_fnc.utils.decorators.time
def intersections(n_lines_file_data_extract, file_data_extract, file_intersec, file_orbit_bundle):
    """Compute intersections between stars and volume defined by a bundle of orbits."""

    if not _os.path.isfile(file_data_extract):
        raise FileNotFoundError(file_data_extract)

    if not _os.path.isfile(file_orbit_bundle):
        raise FileNotFoundError(file_orbit_bundle)

    _clib.intersections(n_lines_file_data_extract,
                        file_data_extract.encode('utf-8'),
                        file_intersec.encode('utf-8'),
                        file_orbit_bundle.encode('utf-8'))


@_fnc.utils.decorators.time
def pre_selection(eps, n_lines_file_data, file_data, file_intersec, file_pre_sel, file_intersec_pre_sel):
    """Pre-select stars with intersection greater than eps."""

    if not _os.path.isfile(file_data):
        raise FileNotFoundError(file_data)

    if not _os.path.isfile(file_intersec):
        raise FileNotFoundError(file_intersec)

    _clib.pre_selection(eps,
                        n_lines_file_data,
                        file_data.encode('utf-8'),
                        file_intersec.encode('utf-8'),
                        file_pre_sel.encode('utf-8'),
                        file_intersec_pre_sel.encode('utf-8'))


def count_file_lines(file_name):
    """Count number lines in a file.

    Note
    ----
    1)Equivalent to bash: wc -l."""

    with open(file_name, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines

#-----------------------------------------------------------------------------

def with_neighbours(ra, dec, angular_radius, min_num_nhb, progress=False):
    """Determine the stars with more or equal number of neighbours
    'min_num_nhb' within an angular radius 'angular_radius'."""

    n = len(ra)
    selected = [False]*n

    for i in _tqdm.tqdm(range(n), ncols=78, disable=not progress):

        ang_d = _invi.misc.angular_distance(ra[i], dec[i], ra, dec)

        ang_d[i] = _np.inf
        sel = ang_d < angular_radius

        if len(sel[sel]) >= min_num_nhb:
            selected[i] = True

    return _np.array(selected)

#-----------------------------------------------------------------------------

def cuts(s_dict, BPRP_orb_est, G_orb_est, prm_gc, intersec):
    #Phase-space selection
    phase_space = intersec > prm_gc['selection']['eps']

    #Cut used when downloading the GDR3
    plx = s_dict['parallax']['parallax'] < prm_gc['selection']['parallax_cut']

    #With measured BPRP colour index
    with_BPRP = _np.logical_not(_np.isnan(BPRP_orb_est))

    #Eliminate the zones with hight foreground contamination
    dec_cut = s_dict['ICRS']['delta'] > prm_gc['selection']['dec_cut']
    ra_cut = s_dict['ICRS']['alpha'] < prm_gc['selection']['ra_cut']
    square_cut = _np.logical_not( _invi.misc.polygon_selection(s_dict['ICRS']['alpha'],
                                                               s_dict['ICRS']['delta'],
                                                               _np.array(prm_gc['selection']['polygon']['square']).T) )

    #HR-Diagram cuts
    main_seq = _invi.misc.polygon_selection(BPRP_orb_est,
                                            G_orb_est,
                                            _np.array(prm_gc['selection']['polygon']['main_seq']).T)

    horizontal_branch = _invi.misc.polygon_selection(BPRP_orb_est,
                                                     G_orb_est,
                                                     _np.array(prm_gc['selection']['polygon']['horizontal_branch']).T)

    giants = _invi.misc.polygon_selection(BPRP_orb_est,
                                          G_orb_est,
                                          _np.array(prm_gc['selection']['polygon']['red_giants']).T)

    #Preliminary selection
    sel = phase_space & plx & with_BPRP & dec_cut & ra_cut & square_cut & (main_seq | horizontal_branch | giants)

    return sel


def final_selection(s_dict, BPRP_orb_est, G_orb_est, prm_gc, intersec):
    #Preliminary selection
    sel = cuts(s_dict, BPRP_orb_est, G_orb_est, prm_gc, intersec)

    sel_nhb = _invi.selection.with_neighbours(s_dict['ICRS']['alpha'][sel],
                                              s_dict['ICRS']['delta'][sel],
                                              prm_gc['selection']['neighbours']['angular_radius'],
                                              prm_gc['selection']['neighbours']['min_num_nhb'])

    final_sel = _np.array([False]*len(sel))
    final_sel[sel] = sel_nhb

    return final_sel

#-----------------------------------------------------------------------------
