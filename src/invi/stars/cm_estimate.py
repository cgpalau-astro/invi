"""Distance estimation from the colour and magnitude of the globular cluster."""

import numpy as _np
import scipy as _scipy

import invi as _invi
import fnc as _fnc

__all_ = ["load_data_isochrone", "distance", "select"]

#-----------------------------------------------------------------------------

def _not_delete_elements(n, elements):
    """Return False for the elements to be deleted. True for the elements to
    be mantained."""
    sel = _np.array([True]*n)
    i = _np.arange(0, n)
    for item in elements:
        sel &= i != item
    return sel


def _define_spline(x, y, types):
    splines = []

    for item in (1, 2, 3):
        sel = types == item
        splines.append( _scipy.interpolate.CubicSpline(x[sel], y[sel], extrapolate=False) )

    return splines


def load_data_isochrone(name_file, verbose=False):
    """Load 'BPRP' and 'G' magnitudes, and splines following the isochrone
    for the main sequence ('type'=1), red-giant ('type'=2), and horizontal
    branch ('type'=3).

    Note
    ----
    1)  This fucntion is especific for the isochone of M68 given in the file:
        '../../data/synthetic_population/M68/isochrone.dat.zip'"""

    #Load CMD37 isochone
    iso = _invi.globular_cluster.synthetic_population._load_CMD_file(name_file,
                                                                     type_file="isochrone",
                                                                     shuffle=False,
                                                                     random_state=None,
                                                                     verbose=verbose)
    #Star types (label)
    types = iso['type']

    #Change type from main-sequence to red-giant branch
    i = [38, 39, 40, 41, 42]
    types[i] = 2

    #Symplify by joining types
    types[types==3] = 2
    types[types==4] = 3
    types[types==7] = 3

    #Delete points that break the strictly increasing sequence and delete types 8 and 9
    n = len(types)
    incr = _not_delete_elements(n, [83, 161, 162, 163, 116, 117, 187, 188, 189, 156])
    sel = incr & (types != 8) & (types != 9)

    #Select data points
    BPRP = iso['BPRP'][sel]
    G = iso['G'][sel]
    types = iso['type'][sel]

    #Invert order type 1 because splines require an strictly increasing sequence
    sel = types == 1
    BPRP[sel] = _np.flip(BPRP[sel])
    G[sel] = _np.flip(G[sel])

    #Definition splines
    splines = _define_spline(BPRP, G, types)

    return {'BPRP': BPRP, 'G': G, 'types': types, 'splines': splines}

#-----------------------------------------------------------------------------

def _BPRP_to_G(types, BPRP, isochrone):
    return isochrone[types-1](BPRP)


def distance(bprp, g, isochrone):
    #Estimated magnitude for each type [mag]
    G1 = _BPRP_to_G(1, bprp, isochrone)
    G2 = _BPRP_to_G(2, bprp, isochrone)
    G3 = _BPRP_to_G(3, bprp, isochrone)

    #Estimated distance [kpc]
    return _np.array([_invi.photometry.magnitudes.m_M_to_d(g, G1),
                      _invi.photometry.magnitudes.m_M_to_d(g, G2),
                      _invi.photometry.magnitudes.m_M_to_d(g, G3)]).T

#Reescriure aquesta funcio:

def select(d, ref, intv):
    """Select the distance 'd' closest to a reference 'ref' within an interval [ref-intv, ref+intv]."""
    n = len(ref)
    d_sel = _np.zeros(n)
    d = _np.nan_to_num(d, nan=_np.inf)

    for i in range(n):
        error = _np.abs(d[i] - ref[i])
        j = _np.argmin(error)

        if _fnc.numeric.within_equal(d[i][j], ref[i]-intv, ref[i]+intv):
            d_sel[i] = d[i][j]
        else:
            d_sel[i] = _np.nan

    d_sel[_np.isinf(d_sel)] = _np.nan

    return d_sel

#-----------------------------------------------------------------------------
