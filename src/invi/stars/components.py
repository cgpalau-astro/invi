"""Classification stream stars."""

import numpy as _np

import invi as _invi
import fnc as _fnc
_sklc = _fnc.utils.lazy.Import("sklearn.cluster")


__all__ = ["classify", "number", "print_number", "main_component"]

#-----------------------------------------------------------------------------

def _normalitzation_data(x, y, z):
    #------------------
    a = x - _np.mean(x)
    b = y - _np.mean(y)
    c = z - _np.mean(z)
    #------------------
    a = a / _np.std(a)
    b = b / _np.std(b)
    c = c / _np.std(c)
    #------------------
    sel = a < 0
    b[sel] = -b[sel]
    c[sel] = -c[sel]
    a[sel] = -a[sel]
    #------------------
    return _np.array([a, b, c]).T


def _clustering_dbscan(x, y, z, eps, min_samples):
    #Normalization data
    data = _normalitzation_data(x, y, z)

    #DBScan clustering
    dbscan = _sklc.DBSCAN(eps=eps, min_samples=min_samples, algorithm="brute").fit(data)
    labels = dbscan.labels_

    #Select outliers
    outliers = labels == -1

    return outliers


def _identification_escapees(prm_gc, s_AAF, not_gc):
    min_samples = prm_gc['stream']['escapees']['min_samples']

    #0 cluster in actions J
    x = s_AAF[3][not_gc]
    y = s_AAF[4][not_gc]
    z = s_AAF[5][not_gc]
    eps_0 = prm_gc['stream']['escapees']['eps_0']
    esc_0 = _clustering_dbscan(x, y, z, eps_0, min_samples)

    #1 cluster in angles A
    x = s_AAF[0][not_gc]
    y = s_AAF[1][not_gc]
    z = s_AAF[2][not_gc]
    eps_1 = prm_gc['stream']['escapees']['eps_1']
    esc_1 = _clustering_dbscan(x, y, z, eps_1, min_samples)

    #Select escapees in one of the methods
    esc = esc_0 | esc_1

    n = len(s_AAF[3])
    escapees = _np.asarray([False]*n)
    escapees[not_gc] = esc

    return escapees


def _sub_streams(prm_gc, time, arm):
    t = -time
    time_intv = prm_gc['stream']['internal']['time_intv']
    return {'s1':  (t>=time_intv[0]) & (t<time_intv[1])  & (arm),
            's2':  (t>=time_intv[1]) & (t<time_intv[2])  & (arm),
            's3':  (t>=time_intv[2]) & (t<time_intv[3])  & (arm),
            'sn': ((t<time_intv[0]) | (t>=time_intv[3])) & (arm)}


def _internal_streams(prm_gc, time, components):
    return {'leading': _sub_streams(prm_gc, time, components['leading']),
            'trailing': _sub_streams(prm_gc, time, components['trailing'])}

#-----------------------------------------------------------------------------

def classify(prm_gc, s_fsr, gc_fsr, gc_aaf, s_aaf):
    """Classification of stars.

    Note
    ----
    1)  gc: globular cluster
        escapees: stars escaped from the progenitor by evaporation, collisions, etc.
        stream: stream stars
        leading: stream stars in the leading arm
        trailing: stream stars in the trailing arm
        internal: internal stream generated during the pericentre passages"""
    #-------------------------------------------------------------------------
    #Definition dictionary
    components = {}

    #Globular Cluster [kpc]
    x = s_fsr[0] - gc_fsr[0]
    y = s_fsr[1] - gc_fsr[1]
    z = s_fsr[2] - gc_fsr[2]

    d = _np.sqrt(x**2.0 + y**2.0 + z**2.0)*1_000.0 #[pc]
    components['gc'] = d < prm_gc['king']['r_truncation']

    not_gc = _np.logical_not(components['gc'])

    #aaf relative to the globular cluster
    s_dgc = _invi.coordinates.aaf_to_dgc(s_aaf, gc_aaf)
    #aaf principal axes stream
    s_AAF = _invi.coordinates.dgc_to_AAF(s_dgc, prm_gc['stream']['varphi'])

    #Escapees
    components['escapees'] = _identification_escapees(prm_gc, s_AAF, not_gc)
    not_escapees = _np.logical_not(components['escapees'])

    #Stream
    components['stream'] = not_escapees & not_gc
    components['leading'] = components['stream'] & (s_AAF[0] > 0.0) #s_AAF[0] = A1
    components['trailing'] = components['stream'] & (s_AAF[0] < 0.0)

    #Time integration
    time = _invi.inverse.integration_time(s_dgc)

    #Internal streams
    components['internal'] = _internal_streams(prm_gc, time, components)

    return components

#-----------------------------------------------------------------------------

def number(components):
    """Number of stars of each component."""
    x = components['stream']
    return {'total': len(x),

            'gc': len(x[components['gc']]),
            'st': len(x[components['stream']]),

            'l': len(x[components['leading']]),
            't': len(x[components['trailing']]),
            'e': len(x[components['escapees']]),

            #Leading arm internal components
            'ls1': len(x[components['internal']['leading']['s1']]),
            'ls2': len(x[components['internal']['leading']['s2']]),
            'ls3': len(x[components['internal']['leading']['s3']]),
            'lsn': len(x[components['internal']['leading']['sn']]),

            #Trailing arm internal components
            'ts1': len(x[components['internal']['trailing']['s1']]),
            'ts2': len(x[components['internal']['trailing']['s2']]),
            'ts3': len(x[components['internal']['trailing']['s3']]),
            'tsn': len(x[components['internal']['trailing']['sn']])}


def print_number(components):
    """Print number stars of each component."""

    n = number(components)

    N = 32
    print("-"*N)

    print(f"Total number stars     = {n['total']:_}")
    print(f"Globular cluster stars = {n['gc']:_}")
    print(f"Stream stars           = {n['st']:_}")

    print("-"*N)

    print(f"Leading arm  = {n['l']:_}")
    print(f"Trailing arm = {n['t']:_}")
    print(f"Escapees     = {n['e']:_}")

    print("-"*N)

    print(f"Leading 1 = {n['ls1']:_}")
    print(f"Leading 2 = {n['ls2']:_}")
    print(f"Leading 3 = {n['ls3']:_}")
    print(f"Leading - = {n['lsn']:_}")

    print("-"*N)

    print(f"Trailing 1 = {n['ts1']:_}")
    print(f"Trailing 2 = {n['ts2']:_}")
    print(f"Trailing 3 = {n['ts3']:_}")
    print(f"Trailing - = {n['tsn']:_}")

    print("-"*N)

    print(f"Leading arm  1+2+3 = {n['ls1']+n['ls2']+n['ls3']:_}")
    print(f"Trailing arm 1+2+3 = {n['ts1']+n['ts2']+n['ts3']:_}")

    print("-"*N)

#-----------------------------------------------------------------------------

def main_component(st_dict, prm_gc):
    """Main component of the stream defined in 'phi' coordinates.

    Note
    ----
    1)  The stream stars are separated in 'main component' and 'cacoon'.
    2)  cacoon = np.logical_not(main_comp)
    3)  A term 'or' is included in the selection of the 'main component'
        because the simulation does not fit the stellar stream and deviates
        down for large phi_1."""

    def within_limits(x, limits):
        return (x > limits[0]) & (x < limits[1])

    #Add unitary spherical radius
    n = len(st_dict['ICRS']['alpha'])
    st_dict['ICRS']['r'] = _np.ones(n)

    #Convert to array
    st_arr = _invi.dicts.dict_to_array(st_dict['ICRS'])

    #Computation phi coordinates
    J = _np.array(prm_gc['stream']['phi']['J_flat'])
    st_arr_phi = _invi.coordinates.ICRS_to_phi.matrix(st_arr, J)
    #st_arr_phi = [phi_r, phi_2, phi_1, ...]
    phi_1 =  st_arr_phi[2] #[deg]
    phi_2 = st_arr_phi[1] #[deg]

    limits = prm_gc['stream']['phi']['main_component']
    #The last term is included because the simulation does not fit the stellar stream and deviates down for large phi_1.
    main_comp = (within_limits(phi_1, limits[0]) & within_limits(phi_2, limits[1])) | within_limits(phi_1, [limits[0][1], 120.0])

    return main_comp

#-----------------------------------------------------------------------------
