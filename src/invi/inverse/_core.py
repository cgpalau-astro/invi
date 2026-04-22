"""Core function used for the Inverse Time Integration (invi) method."""

import copy as _copy
import numpy as _np

import invi as _invi

__all__ = ["integration_time", "integration", "integration_general_potential"]

#-----------------------------------------------------------------------------

def integration_time(s_dgc):
    #Modulus and integration time: s_dgc = [A_i, J_i, F_i] i=(r, phi, z)
    mod_A = _np.sqrt( s_dgc[0]**2.0 + s_dgc[1]**2.0 + s_dgc[2]**2.0 ) #[rad]
    mod_F = _np.sqrt( s_dgc[6]**2.0 + s_dgc[7]**2.0 + s_dgc[8]**2.0 ) #[rad/Myr]
    time = mod_A/mod_F #[Myr]
    return time


def integration(s_dgc):
    #Integration time
    time = integration_time(s_dgc) #[Myr]

    #Inverse time integration: alpha = A - F*time
    #s_dgc = [A_i, J_i, F_i] i=(r, phi, z)
    alpha_r   = s_dgc[0] - s_dgc[6] * time #[rad]
    alpha_phi = s_dgc[1] - s_dgc[7] * time
    alpha_z   = s_dgc[2] - s_dgc[8] * time

    return _np.array([alpha_r, alpha_phi, alpha_z])

#-----------------------------------------------------------------------------

def integration_general_potential(x, sample_icrs, prm_gc, prm):
    #Parameters 'invi' for aaf computations
    accuracy = prm_gc['potential_estimation']['accuracy']
    b, maxn, tintJ, ntintJ = prm['isochrone_approx'][accuracy].values() #POTSER HAURIA D'ANAR DINS D'OPCIONS
    #---------------------------------------------------
    #Copy dictionary with Milky Way potential parameters
    prm_mw = _copy.deepcopy(prm['mw'])

    #Initialisation parameters potential
    prm_mw['halo']['q'] = prm['mw']['halo']['q']*x[0]
    prm_mw['disc']['M'] = prm['mw']['disc']['M']*x[1]
    prm_mw['disc']['a'] = prm['mw']['disc']['a']*x[2]
    prm_mw['halo']['a'] = prm['mw']['halo']['a']*x[3]

    #Definition Milky Way potential
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm_mw)
    #---------------------------------------------------
    #Position M68 ICRS
    gc_icrs = _invi.dicts.dict_to_array(prm_gc['ICRS'])

    #Position M68 in FSR [kpc, kpc/Myr]
    gc_fsr = _invi.coordinates.ICRS_to_FSR(gc_icrs, prm['sun'], mw.v_lsr)

    #Angle, actions, and frequencies globular cluster
    gc_aaf = _invi.coordinates.FSR_to_aaf(gc_fsr, mw.potential, b, maxn, tintJ, ntintJ)
    #---------------------------------------------------
    #Sample in FSR[kpc, kpc/Myr]
    sample_fsr = _invi.coordinates.ICRS_to_FSR(sample_icrs, prm['sun'], mw.v_lsr)

    #Compute angles [rad], actions [kpc^2/Myr], and frequencies [rad/Myr]
    sample_aaf = _invi.coordinates.FSR_to_aaf(sample_fsr, mw.potential, b, maxn, tintJ, ntintJ)
    #---------------------------------------------------
    #Action, angle and frequency relative to the globular cluster
    sample_dgc = _invi.coordinates.aaf_to_dgc(sample_aaf, gc_aaf)

    #Inverse time integration
    sample_alpha = _invi.inverse.integration(sample_dgc)
    #---------------------------------------------------
    return sample_dgc, sample_alpha

#-----------------------------------------------------------------------------
