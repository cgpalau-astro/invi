"""Computation angle-action-frequencies using agama."""

import numpy as _np

import fnc as _fnc
_agama = _fnc.utils.lazy.Import("agama")

import invi.units as _un

__all__ = ["staeckel_fudge"]

_agama.setUnits(mass=_un.u.M, length=_un.u.L, velocity=_un.u.V)

#-----------------------------------------------------------------------------
#Stäeckel Fudge

def _aaf_staeckel_fudge(action_finder, w_fsr):
    """Evaluate angles, actions, and frequencies from the action_finder."""
    #------------------------------------------------------------
    w_galpy = _un.galactic_to_galpy(w_fsr)
    actions, angles, freq = action_finder(w_galpy.T, angles=True)
    #------------------------------------------------------------
    #Galpy units to galactic:
    #[u.L*u.V, rad, rad/u.T] -> [kpc^2/Myr, rad, rad/Myr]
    Jr = _un.uL2invuT_to_kpc2invMyr( actions.T[0] )
    Jphi = _un.uL2invuT_to_kpc2invMyr( actions.T[2] )
    Jz = _un.uL2invuT_to_kpc2invMyr( actions.T[1] )

    Ar = angles.T[0]
    Aphi = angles.T[2]
    Az = angles.T[1]

    Fr = _un.invuT_to_invMyr( freq.T[0] )
    Fphi = _un.invuT_to_invMyr( freq.T[2] )
    Fz = _un.invuT_to_invMyr( freq.T[1] )
    #------------------------------------------------------------
    return Ar, Aphi, Az, Jr, Jphi, Jz, Fr, Fphi, Fz

def staeckel_fudge(w_fsr, potential_agama):
    """Defines action_finder using the Stäeckel fudge and evaluate angles,
    actions, and frequencies.

    Note
    ----
    1)  Angles: [rad]
        Actions: [kpc^2/Myr]
        Frequencies: [rad/Myr]"""

    action_finder = _agama.ActionFinder(potential_agama, interp=False)
    aaf = _aaf_staeckel_fudge(action_finder, w_fsr)
    return _np.array(aaf)

#-----------------------------------------------------------------------------
"""
def staeckel_fudge_mean_orbit(w_car, potential, T, N, accuracy=1.0E-15, verbose=True):

    Compute angle, actions and frequencies using the Staeckel Fudge method for
    n particles along a orbit computed from their phase-space position and
    return the initial angle and the mean action and frequency.

    Parameters
    ----------
    w_car : np.array
        Phase-space positions in cartesian coordinates and natural units:
        np.shape(ic_fsr) = (6,n)
        [kpc, km/s]
    potential : agama.Potential
        Potential in AGAMA units
    T : float
        Integration time [Myr]
    N : int
        Number of stored points from the orbit
    accuracy : float
        Accuracy orbit integration
    verbose : bool
        Print progress bar aaf determination

    Returns
    -------
    class
        Actions, angles and frequencies in natural units
        [rad, kpc^2/Myr, rad/Myr]

    #---------------------------------------------------------------------
    #Compute orbit for the stream stars: Time [Myr], Orbit [kpc, km/s]
    #Compute Action Angle Frequencies: [rad, kpc^2/Myr, rad/Myr]
    #---------------------------------------------------------------------
    #Define AGAMA Action-Finder
    action_finder = agama.ActionFinder(potential, interp=False)
    #---------------------------------------------------------------------
    #Estimate memory required. Maximum: 10.0 GiB
    n = len(w_car[0])
    mem_B = estimated_memory_array(n*6*N*2)
    mem_GiB = mem_B / 1024**3.0
    #---------------------------------------------------------------------
    aaf = [[]]*n
    if mem_GiB < 10.0:
        #Compute orbit and aaf using n cores:
        #-----------------------------------------------------------------
        t, w_fsr = orbit_agama.orbit(w_car, potential, T, N, accuracy=accuracy)
        #-----------------------------------------------------------------
        for i in tqdm(range(n), disable=not verbose):
            aaf[i] = genc.AAFMean( staeckel_fudge(w_fsr[i], action_finder) ).wf
        aaf = np.asarray(aaf).T
        #-----------------------------------------------------------------
    else:
        #Compute orbit using 1 core and aaf using n cores:
        #-----------------------------------------------------------------
        mem_available_B = psutil.virtual_memory()[1]
        cprint("INFO: Memory required / available:"
                f" ~{memory_human_readable(mem_B)} / {memory_human_readable(mem_available_B)}"
                " -> Using one CPU core to calculate the orbits.", "green")
        #-----------------------------------------------------------------
        for i in tqdm(range(n), disable=not verbose):
            t, w_fsr = orbit_agama.orbit(w_car.T[i], potential, T, N, accuracy=accuracy)
            aaf[i] = genc.AAFMean( staeckel_fudge(w_fsr, action_finder) ).wf
        aaf = np.asarray(aaf).T
        #-----------------------------------------------------------------
    #---------------------------------------------------------------------
    return genc.AAF(aaf)
"""
#-----------------------------------------------------------------------------
