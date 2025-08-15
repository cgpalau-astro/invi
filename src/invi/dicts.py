"""Definitions of dictionaries."""

import numpy as _np

import invi as _invi

__all__ = ["dict_to_array", "array_to_dict", "phase_space", "aaf", "alpha"]

#-----------------------------------------------------------------------------

def dict_to_array(xdict):
    """Returns values of a dict as numpy.array given a list of allowed keys."""
    #-------------------------------------------------------
    def get_values(xdict, list_keys):
        """Returns values of a dict given a list of keys."""
        return [xdict[item] for item in list_keys]
    #-------------------------------------------------------
    keys = set(xdict.keys())

    #List of allowed keys
    icrs = ['r', 'delta', 'alpha', 'mu_r', 'mu_delta', 'mu_alpha_str']
    car = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']
    cyl = ['R', 'phi', 'z', 'v_R', 'v_phi', 'v_z']
    sph = ['r', 'theta', 'phi', 'v_r', 'v_theta', 'v_phi']
    aaf = ['Ar', 'Aphi', 'Az', 'Jr', 'Jphi', 'Jz', 'Fr', 'Fphi', 'Fz'] #Equal than 'dgc'
    AAF = ['A1', 'A2', 'A3', 'J1', 'J2', 'J3', 'F1', 'F2', 'F3']
    alpha = ['a_r', 'a_phi', 'a_z']
    ALPHA = ['A_1', 'A_2', 'A_3']

    for item in [icrs, car, cyl, sph, aaf, AAF, alpha, ALPHA]:
        if set(item).issubset(keys):
            return _np.array(get_values(xdict, item))

    raise KeyError(f"{xdict.keys()} not in the list of allowed keys.")


def array_to_dict(arr, type_keys):
    """Returns a numpy.array as a dict with keys defined by type_keys."""
    match type_keys:
        case 'ICRS':
            keys = ['r', 'delta', 'alpha', 'mu_r', 'mu_delta', 'mu_alpha_str']
        case 'car':
            keys = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']
        case 'cyl':
            keys = ['R', 'phi', 'z', 'v_R', 'v_phi', 'v_z']
        case 'sph':
            keys = ['r', 'theta', 'phi', 'v_r', 'v_theta', 'v_phi']
        case 'aaf':
            keys = ['Ar', 'Aphi', 'Az', 'Jr', 'Jphi', 'Jz', 'Fr', 'Fphi', 'Fz']
        case 'dgc':
            keys = ['Ar', 'Aphi', 'Az', 'Jr', 'Jphi', 'Jz', 'Fr', 'Fphi', 'Fz']
        case 'AAF':
            keys = ['A1', 'A2', 'A3', 'J1', 'J2', 'J3', 'F1', 'F2', 'F3']
        case 'alpha':
            keys = ['a_r', 'a_phi', 'a_z']
        case 'ALPHA':
            keys = ['A_1', 'A_2', 'A_3']
        case _:
            raise KeyError("Allowed type_keys: {'ICRS', 'car', 'cyl', 'sph', 'aaf', 'dgc', 'AAF', 'alpha', 'ALPHA'}")

    return dict((keys[i], arr[i]) for i in range(len(arr)))

#-----------------------------------------------------------------------------

def _ICRS(w_icrs):
    #ICRS coordinates [kpc, deg, deg, km/s, mas/yr, mas/yr]
    icrs = array_to_dict(w_icrs, 'ICRS')
    return icrs


def _FSR(w_fsr):
    #Galactic coordinates [kpc, kpc/Myr]
    w_car = w_fsr
    w_cyl = _invi.coordinates.car_to_cyl(w_fsr)
    w_sph = _invi.coordinates.car_to_sph(w_fsr)

    return {'car': array_to_dict(w_car, 'car'),
            'cyl': array_to_dict(w_cyl, 'cyl'),
            'sph': array_to_dict(w_sph, 'sph')}


def phase_space(w_fsr, w_icrs):
    return {'FSR': _FSR(w_fsr), 'ICRS': _ICRS(w_icrs)}

#-----------------------------------------------------------------------------

def aaf(s_aaf, gc_aaf, varphi):
    #Relative to the globular cluster
    s_dgc = _invi.coordinates.aaf_to_dgc(s_aaf, gc_aaf)

    #Rotation stream principal axes
    s_AAF = _invi.coordinates.dgc_to_AAF(s_dgc, varphi)

    return {'aaf': array_to_dict(s_aaf, 'aaf'),
            'dgc': array_to_dict(s_dgc, 'dgc'),
            'AAF': array_to_dict(s_AAF, 'AAF')}

#-----------------------------------------------------------------------------

def alpha(s_alpha, varphi):
    s_ALPHA = _invi.coordinates.alpha_to_ALPHA(s_alpha, varphi)
    return {'alpha': _invi.dicts.array_to_dict(s_alpha, 'alpha'),
            'ALPHA': _invi.dicts.array_to_dict(s_ALPHA, 'ALPHA')}

#-----------------------------------------------------------------------------
