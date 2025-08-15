"""Computation angle-action-frequencies using galpy."""

import os as _os
from multiprocessing import Pool as _Pool

import numpy as _np

import invi as _invi

import fnc as _fnc
_galpy_orbit = _fnc.utils.lazy.Import("galpy.orbit")
_galpy_actionAngle = _fnc.utils.lazy.Import("galpy.actionAngle")


__all__ = ["scale_factor",
           "staeckel_fudge", "isochrone_approx",
           "torus_mapper",
           "hessian", "eig_hessian"]

#-----------------------------------------------------------------------------

def scale_factor(w_fsr, potential_galpy, T, N, method):
    """Determine the scale factors for the methods: 'estimateDeltaStaeckel'
    and 'estimateBIsochrone'.

    Parameters
    ----------
    method : {estimateDeltaStaeckel | estimateBIsochrone}

    Returns
    -------
    'delta' for estimateDeltaStaeckel
    '[b_min, b_median, b_max]' over the orbit for estimateBIsochrone

    Note
    ----
    1)  from galpy.actionAngle import estimateDeltaStaeckel, estimateBIsochrone"""
    #-----------------------------------------------------------------
    w_cyl_galpy = _invi.coordinates.car_to_cyl_galpy(w_fsr)
    t_galpy = _np.linspace(0.0, _invi.units.Myr_to_uT(T), N)
    orb_galpy = _galpy_orbit.Orbit(w_cyl_galpy)
    orb_galpy.integrate(t_galpy, potential_galpy, method="symplec6_c")
    #-----------------------------------------------------------------
    return method(potential_galpy, orb_galpy.R(t_galpy), orb_galpy.z(t_galpy))

#-----------------------------------------------------------------------------
#Stäeckel Fudge

def _aaf_staeckel_fudge(w_fsr, action_finder):
    """Evaluate angles, actions, and frequencies from the action_finder."""

    w_cyl_galpy = _invi.coordinates.car_to_cyl_galpy(w_fsr)
    orb_galpy = _galpy_orbit.Orbit(w_cyl_galpy)

    #Action - Frequency - Angles in galpy units
    afa_galpy = action_finder.actionsFreqsAngles(orb_galpy)

    #From galpy to galactic units
    aaf = _invi.units.afa_galpy_to_aaf(afa_galpy)

    return aaf


def staeckel_fudge(w_fsr, potential_galpy, delta):
    """Defines action_finder using the Stäeckel fudge and evaluate angles,
    actions, and frequencies.

    Note
    ----
    1)  Angles: [rad]
        Actions: [kpc^2/Myr]
        Frequencies: [rad/Myr]"""

    #Define action finder (c implementation True)
    action_finder = _galpy_actionAngle.actionAngleStaeckel(pot=potential_galpy, delta=delta, c=True)
    aaf = _aaf_staeckel_fudge(w_fsr, action_finder)
    return _np.array(aaf)

#-----------------------------------------------------------------------------
#Isochrone Approximation

def _isochrone_approx_singlecore(w_fsr, action_finder, maxn):
    """Evaluate angles, actions, and frequencies from the action_finder."""

    w_cyl_galpy = _invi.coordinates.car_to_cyl_galpy(w_fsr)

    #j_r,lz,j_z, O_r,O_phi,O_z, angle_r,angle_phi,angle_z
    afa_galpy = action_finder.actionsFreqsAngles(*w_cyl_galpy, maxn=maxn)

    #From galpy to galactic units
    aaf = _invi.units.afa_galpy_to_aaf(afa_galpy)

    return aaf


class _Init_ia:
    def __init__(self, w_fsr, maxn, action_finder):
        self.w_fsr = w_fsr
        self.maxn = maxn
        self.action_finder = action_finder


def _isochrone_approx_multicore(Init):
    return _isochrone_approx_singlecore(Init.w_fsr, Init.action_finder, maxn=Init.maxn)


def isochrone_approx(w_fsr, potential_galpy, b, maxn, tintJ, ntintJ, n_cpu=None, progress=False):
    """Defines action_finder using the Isochrone Approximation and evaluate
    angles, actions, and frequencies using n_cpu.

    Note
    ----
    1)  Angles: [rad]
        Actions: [kpc^2/Myr]
        Frequencies: [rad/Myr]

    Docs:
    -----
    1)  https://docs.galpy.org/en/v1.10.1/actionAngle.html
    2)  https://docs.galpy.org/en/v1.10.1/reference/aaisochroneapprox.html

    Parameters:
    ----------
    b : Improves accuracy of J_r and J_z. It has to be determined for each
        orbit. For the mw potential and m68: b=0.622.

    maxn : For maxn<=4 it does not present resonance trapping at F_z/F_r=3/4.
           It presents small deviations from the tendency for ntintJ~50_000.0
           Myr, but not for ntintJ<~20_000.0.
           For maxn>=5 it presents resonance trapping at F_z/F_r=3/4. This is
           independent of b and tintJ.

    tintJ : Integration time. Larger values increase accuracy of aaf,
            specially J_r and J_z and computational time. Default:
            tintJ=3_555.60807862529 Myr (equivalent to 100.0 u.T). The Default
            frequencies can be improved with tintJ=5_000 Myr.
            Better accuracy: tintJ=50_000.0 Myr.

    ntintJ : Number of evaluations. Default: ntintJ=10_000. The accuracy of
             the aaf do not improve for larger values, but the computational
             time increases."""
    #-------------------------------------------------------------------------
    action_finder = _galpy_actionAngle.actionAngleIsochroneApprox(pot=potential_galpy,
                                                                  b=b,
                                                                  integrate_method="symplec4_c",
                                                                  tintJ=_invi.units.Myr_to_uT(tintJ),
                                                                  ntintJ=ntintJ)

    #Determine number of cores
    if n_cpu is None:
        n_cpu = _os.cpu_count()

    n = len(w_fsr.T)

    if _np.shape(w_fsr) == (6,):
        output = _isochrone_approx_singlecore(w_fsr, action_finder, maxn=maxn)
    else:
        init = [[]]*n
        for i in range(n):
            init[i] = _Init_ia(w_fsr.T[i], maxn, action_finder)
        output = _fnc.utils.pool.run(_isochrone_approx_multicore, init, n_cpu, progress)
    #-------------------------------------------------------------------------
    return _np.asarray(output).T

#-----------------------------------------------------------------------------
#Torus mapper

def _torus_mapper_singlecore(w_aaf, w_finder, tol):
    #[A_i, J_i, F_i] -> [J_i, F_i, A_i] galpy units
    w_afa_galpy = _invi.units.aaf_to_afa_galpy(w_aaf)

    #[J_i, A_i]
    #J_i has to be passed as float and A_i as np.array
    w_aa_galpy = (w_afa_galpy[0][0], w_afa_galpy[1][0], w_afa_galpy[2][0],
                  w_afa_galpy[6], w_afa_galpy[7], w_afa_galpy[8])

    w_cyl_galpy, _, _, _, _error_int = w_finder.xvFreqs(*w_aa_galpy, tol=tol)

    w_fsr = _invi.coordinates.cyl_galpy_to_car(w_cyl_galpy[0])
    return w_fsr


class _Init_tm:
    def __init__(self, w_aaf, tol, w_finder):
        self.w_aaf = w_aaf
        self.tol = tol
        self.w_finder = w_finder


def _torus_mapper_multicore(Init):
    return _torus_mapper_singlecore(Init.w_aaf, Init.w_finder, tol=Init.tol)


def torus_mapper(w_aaf, potential_galpy, tol=1.0E-3, n_cpu=None, progress=True):
    """Torus mapper.

    Web
    ---
    1)  https://docs.galpy.org/en/v1.10.1/actionAngle.html
    2)  https://github.com/PaulMcMillan-Astro/Torus"""
    #-------------------------------------------------------------------------
    w_finder = _galpy_actionAngle.actionAngleTorus(pot=potential_galpy)

    #Determine number of cores
    if n_cpu is None:
        n_cpu = _os.cpu_count()

    n = len(w_aaf.T)

    if _np.shape(w_aaf) == (9,):
        output = _torus_mapper_singlecore(w_aaf, w_finder, tol=tol)
    else:
        init = [[]]*n
        for i in range(n):
            init[i] = _Init_tm(w_aaf.T[i], tol, w_finder)
        output = _fnc.utils.pool.run(_torus_mapper_multicore, init, n_cpu, progress)
    #-------------------------------------------------------------------------
    return _np.asarray(output).T

#-----------------------------------------------------------------------------

def hessian(w_aaf, potential_galpy, tol=1E-3):
    """Compute the Hessian matrix (H=F/J) in rad/kpc^2 at
    the position w_aaf using the Torus-Mapper method.

    Note
    ----
    1) tol=2E-4 returns an error code but more accurate results."""

    #Definition torus mapper
    torus_mapper = _galpy_actionAngle.actionAngleTorus(pot=potential_galpy)

    #[A_i, J_i, F_i] -> [J_i, F_i, A_i] galpy units
    w_afa_galpy = _invi.units.aaf_to_afa_galpy(w_aaf)
    #Actions have to be passed as floats
    actions_galpy = (w_afa_galpy[0][0], w_afa_galpy[1][0], w_afa_galpy[2][0])

    #Hessian matrix [rad/u.L^2]
    H_galpy, _, _, _, _error_int = torus_mapper.hessianFreqs(*actions_galpy, tol=tol)

    #Hessian matrix [rad/kpc^2]
    H = _invi.units.invuL_to_invkpc(_invi.units.invuL_to_invkpc(H_galpy))

    return H


def eig_hessian(w_aaf, potential_galpy, tol=1E-3):
    """Compute the eigenvalues of the Hessian matrix (H=F/J) in rad/kpc^2 at
    the position w_aaf using the Torus-Mapper method.

    Note
    ----
    1) tol=2E-4 returns an error code but more accurate results."""

    #Hessian matrix [rad/kpc^2]
    H = hessian(w_aaf, potential_galpy, tol)

    #Diagonalisation
    eig = _np.array(sorted(_np.linalg.eigvals(H), key=abs, reverse=True))

    return eig

#-----------------------------------------------------------------------------
