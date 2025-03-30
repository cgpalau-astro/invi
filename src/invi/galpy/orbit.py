"""Compute orbits using a galpy potential."""

import numpy as _np

import invi.units as _un
import invi.coordinates as _co

from fnc.utils import lazy as _lazy
_galpy_orbit = _lazy.Import("galpy.orbit")

__all__ = ["integrate"]

#-----------------------------------------------------------------------------

def integrate(ic_fsr, potential_galpy, T, N):
    """Integrate orbit using galpy.

    Parameters
    ----------
    ic_fsr : _np.array
        Initial conditions for 1 orbit in Galactic units [kpc, kpc/Myr]
        numpy.shape(ic_fsr) = (6,)
    potential_galpy : galpy.potential
        Galpy potential in galpy units [u.U].
    T : float
        Integration time [Myr].
    N : int
        Number stored points.

    Returns
    -------
    np.array
        Time [Myr].
    np.array
        orbit [kpc, kpc/Myr]
        numpy.shape(w_fsr) = (6, N)"""
    #----------------------------------------------
    shape = _np.shape(ic_fsr)

    if len(shape) == 1:
        if shape[0] != 6:
            raise ValueError(f"np.shape(ic_fsr) = {shape} != (6,)")
    else:
        raise ValueError(f"np.shape(ic_fsr) = {shape} != (6,)")
    #----------------------------------------------
    #Convert from cartesian to cylindrical coordinates in galpy units and format
    ic_cyl_galpy = _co.car_to_cyl_galpy(ic_fsr)

    #Definition time in galpy units
    t_galpy = _np.linspace(0.0, _un.Myr_to_uT(T), N)
    #----------------------------------------------
    #Integrate orbit
    orbit = _galpy_orbit.Orbit(ic_cyl_galpy)
    orbit.integrate(t_galpy, potential_galpy, method="symplec6_c")
    #Obtain cartesian coordinates [u.L]
    w_car_galpy = _np.array([orbit.x(t_galpy), orbit.y(t_galpy), orbit.z(t_galpy),
                             orbit.vx(t_galpy), orbit.vy(t_galpy), orbit.vz(t_galpy)])
    #----------------------------------------------
    #Convert from galpy to galactic units
    t = _un.uT_to_Myr(t_galpy)
    w_fsr = _un.galpy_to_galactic(w_car_galpy)
    #----------------------------------------------
    return t, w_fsr

#-----------------------------------------------------------------------------
