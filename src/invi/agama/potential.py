"""Definition of a class including a agama potential of the Milky Way and the
circular velocity at the position of the Sun."""

import numpy as _np

from fnc.utils import lazy as _lazy
_agama = _lazy.Import("agama")

import invi.units as _un

__all__ = ["DefineGalaxy"]

_agama.setUnits(mass=_un.u.M, length=_un.u.L, velocity=_un.u.V)

#-----------------------------------------------------------------------------

def _potential_galaxy(mw):
    """Define an agama potential of a Milky Way like galaxy. This model is
    based on BovyMWPotential2014 but allows general parameters.

    Parameters
    ----------
    mw : dict
        Parameters characterising a BovyMWPotential2014-like potential in
        Galactic units.

    Returns
    -------
    potential : agama.potential
        Agama potential in galpy units."""
    #--------------------------------------------
    #From galactic to galpy units

    #Bulge:
    rho_b = _un.Msuninvkpc3_to_uMinvuL3(mw['bulge']['rho'])
    a_b = _un.kpc_to_uL(mw['bulge']['a'])
    rc_b = _un.kpc_to_uL(mw['bulge']['rc'])
    gamma_b = mw['bulge']['gamma']
    beta_b = mw['bulge']['beta']
    chi_b = mw['bulge']['chi']

    #Disc: Miyamoto-Nagai
    M_d = _un.Msun_to_uM(mw['disc']['M'])
    a_d = _un.kpc_to_uL(mw['disc']['a'])
    b_d = _un.kpc_to_uL(mw['disc']['b'])

    #Dark Halo: NFW
    rho_h = _un.Msuninvkpc3_to_uMinvuL3(mw['halo']['rho'])
    a_h = _un.kpc_to_uL(mw['halo']['a'])
    gamma_h = mw['halo']['gamma']
    beta_h = mw['halo']['beta']
    q_h = mw['halo']['q']
    #--------------------------------------------
    bulge = _agama.Potential(type="Spheroid", densityNorm=rho_b, gamma=gamma_b, beta=beta_b, scaleRadius=a_b, outerCutoffRadius=rc_b, cutoffStrength=chi_b)

    disc = _agama.Potential(type="MiyamotoNagai", mass=M_d, scaleRadius=a_d, scaleHeight=b_d)

    #rho_dh/q_dh to keep mass independent of q_dh
    halo = _agama.Potential(type="Spheroid", densityNorm=rho_h/q_h, gamma=gamma_h, beta=beta_h, scaleRadius=a_h, axisRatioZ=q_h)

    potential = _agama.Potential(bulge, disc, halo)
    #--------------------------------------------
    return potential

#-----------------------------------------------------------------------------

def _circular_velocity(R_sun, potential_agama):
    """Returns the circular velocity [kpc/Myr] at the position of the Sun [kpc]
    of an agama potential."""
    Rsun = _un.kpc_to_uL(R_sun)
    xyz = _np.column_stack((Rsun, Rsun*0.0, Rsun*0.0))
    vc = _un.uV_to_kpcMyr(_np.sqrt(-Rsun * potential_agama.force(xyz)[:,0])[0])
    return vc

#-----------------------------------------------------------------------------

class DefineGalaxy:
    """Defines a Milky Way galaxy with an agama potential and the circular
    velocity [kpc/Myr] at the position of the Sun [kpc]."""
    def __init__(self, R_sun, parameters_potential):
        #Galactic potential in galpy units [u.M, u.L]
        self.potential = _potential_galaxy(parameters_potential)
        #Circular velocity [kpc/Myr] at R_sun [kpc]
        self.v_lsr = _circular_velocity(R_sun, self.potential)

#-----------------------------------------------------------------------------
