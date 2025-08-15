"""Compute orbits using an agama potential."""

import numpy as _np

import invi.units as _un
import invi.coordinates as _co

from fnc.utils import lazy as _lazy
_agama = _lazy.Import("agama")

__all__ = ["integrate"]

_agama.setUnits(mass=_un.u.M, length=_un.u.L, velocity=_un.u.V)

#-----------------------------------------------------------------------------

def integrate(ic_fsr, potential_agama, T, N, accuracy=1.0E-15):
    """Integrate orbits using agama.

    Parameters
    ----------
    ic_fsr : np.array
        Initial conditions for n orbits [kpc, kpc/Myr]: shape(ic_fsr) = (6,n)
    potential_agama : agama.potential
        Agama potential [u.U].
    T : float
        Integration time [Myr].
    N : int
        Number stored points.
    accuracy : float
        Accuracy integration.

    Returns
    -------
    np.array
        Time [Myr].
    np.array
        n orbits [kpc, kpc/Myr]: shape(w_fsr) = (n, 6, N)"""
    #--------------------------------------------------
    shape = _np.shape(ic_fsr)

    if len(shape) == 1:
        if shape[0] != 6:
            raise ValueError(f"np.shape(ic_fsr) = {shape} != (6,)")
    else:
        if shape[0] != 6:
            raise ValueError(f"np.shape(ic_fsr) = {shape} != (6,n)")
    #--------------------------------------------------
    #Convert from galactic to galpy units
    ic_galpy = _un.galactic_to_galpy(ic_fsr)
    T_galpy = _un.Myr_to_uT(T)

    #Compute orbit
    orbit = _agama.orbit(potential=potential_agama,
                         ic=ic_galpy.T,
                         time=T_galpy,
                         trajsize=N,
                         accuracy=accuracy)
    #--------------------------------------------------
    #Convert from galpy to galactic units
    if _np.shape(orbit) == (2,):
        t_galpy = orbit[0]
        t = _un.uT_to_Myr(t_galpy)

        w_galpy = orbit[1]
        w_fsr = _un.galpy_to_galactic(w_galpy.T)
    else:
        t_galpy = orbit[0][0]
        t = _un.uT_to_Myr(t_galpy)

        n = len(orbit)
        w_fsr = [[]]*n
        for i in range(n):
            w_galpy = orbit[i][1]
            w_fsr[i] = _un.galpy_to_galactic(w_galpy.T)
        w_fsr = _np.asarray(w_fsr)
    #--------------------------------------------------
    return t, w_fsr

#-----------------------------------------------------------------------------
