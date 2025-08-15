"""Functions to compute the orbit of a globular cluster."""

import numpy as _np

import invi as _invi

__all__ = ["frw", "bck_frw", "ic_FSR_aaf"]

#-----------------------------------------------------------------------------

def frw(prm_gc, prm, T, N):
    """Compute an orbit forwards given the initial condition of a globular
    cluster."""

    #Milky Way potential (galpy)
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Initial condition
    ic_icrs = _invi.dicts.dict_to_array(prm_gc['ICRS'])

    ic_fsr = _invi.coordinates.ICRS_to_FSR(ic_icrs, prm['sun'], mw.v_lsr)

    #Dictionary including the initial condition
    ic = _invi.dicts.phase_space(ic_fsr, ic_icrs)

    #Forward orbit
    t, w_fsr = _invi.galpy.orbit.integrate(ic_fsr, mw.potential, T, N)

    w_icrs = _invi.coordinates.FSR_to_ICRS(w_fsr, prm['sun'], mw.v_lsr)

    #Dictionary including the orbit
    orbit = _invi.dicts.phase_space(w_fsr, w_icrs)

    return {'ic': ic, 't': t, 'orbit': orbit}

#-----------------------------------------------------------------------------

#hauria de canviar els noms de les variables a tot el fitxer.

def bck_frw(prm_gc, prm):
    """Compute an orbit backwards and forwards given the initial condition of
    a globular cluster. The data returned corresponds to the orbit forwards."""

    #Milky Way potential (galpy)
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Orbit parameters
    T = prm_gc['orbit']['T']
    N = prm_gc['orbit']['N']

    #Initial condition bck
    w_icrs_bck = _invi.dicts.dict_to_array(prm_gc['ICRS'])

    ic_fsr_bck = _invi.coordinates.ICRS_to_FSR(w_icrs_bck, prm['sun'], mw.v_lsr)

    #Inverse orbit bck
    _t_bck, w_fsr_bck = _invi.galpy.orbit.integrate(ic_fsr_bck, mw.potential, -T, N)

    #Initial condition frw
    ic_fsr_frw = w_fsr_bck.T[-1]
    ic_icrs_frw = _invi.coordinates.FSR_to_ICRS(ic_fsr_frw, prm['sun'], mw.v_lsr)

    #Dictionary including the initial condition
    ic = _invi.dicts.phase_space(ic_fsr_frw, ic_icrs_frw)

    #Forward orbit
    t_frw, w_fsr_frw = _invi.galpy.orbit.integrate(ic_fsr_frw, mw.potential, T, N)

    w_icrs_frw = _invi.coordinates.FSR_to_ICRS(w_fsr_frw, prm['sun'], mw.v_lsr)

    #Dictionary including the orbit frw
    orbit = _invi.dicts.phase_space(w_fsr_frw, w_icrs_frw)

    return {'ic': ic, 't': t_frw, 'orbit': orbit}

#-----------------------------------------------------------------------------

def ic_FSR_aaf(prm_gc, prm):
    """Initial conditions in FSR and aaf."""

    #Definition Milky Way potential
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Position M68 ICRS
    gc_icrs = _invi.dicts.dict_to_array(prm_gc['ICRS'])

    #Position M68 in FSR [kpc, kpc/Myr]
    gc_fsr = _invi.coordinates.ICRS_to_FSR(gc_icrs, prm['sun'], mw.v_lsr)

    #Parameters accuracy
    b, maxn, tintJ, ntintJ = prm['isochrone_approx']['accuracy'].values()

    #Angle, actions, and frequencies globular cluster
    gc_aaf = _invi.coordinates.FSR_to_aaf(gc_fsr, mw.potential, b, maxn, tintJ, ntintJ)

    return gc_fsr, gc_aaf

#-----------------------------------------------------------------------------
