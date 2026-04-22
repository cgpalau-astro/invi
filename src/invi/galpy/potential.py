"""Definition of a class including a galpy potential of the Milky Way and the
circular velocity at the position of the Sun. It is also included a function to
obtain the parameters of the potentials for the petar code."""

import numpy as _np

from fnc.utils import lazy as _lazy
_gl = _lazy.Import("galpy")
_gl_pot = _lazy.Import("galpy.potential")

import invi.units as _un

__all__ = ["n_args", "DefineGalaxy", "DefineSpherical"]

#-----------------------------------------------------------------------------

def n_args(potential_galpy):
    """Return number and arguments of a galpy potential."""
    return _gl.orbit.integrateFullOrbit._parse_pot(potential_galpy)

#-----------------------------------------------------------------------------

def _circular_velocity(R_sun, potential_galpy):
    """Returns the circular velocity [kpc/Myr] at the position of the Sun
    [kpc] of a galpy potential."""
    Rsun = _un.kpc_to_uL(R_sun)
    vc = _un.uV_to_kpcMyr(_gl_pot.vcirc(potential_galpy, Rsun))
    return vc

#-----------------------------------------------------------------------------

def _potential_galaxy(mw):
    """Define a galpy potential of a Milky Way like galaxy. This model is
    based on BovyMWPotential2014 but allows general parameters.

    Parameters
    ----------
    mw : dict
        Parameters characterising a BovyMWPotential2014-like potential in
        Galactic units.

    Returns
    -------
    potential : galpy.potential
        Galpy potential in galpy units."""
    #--------------------------------------------
    #From galactic to galpy units

    #Bulge:
    rho_b = _un.Msuninvkpc3_to_uMinvuL3(mw['bulge']['rho'])
    a_b = _un.kpc_to_uL(mw['bulge']['a'])
    rc_b = _un.kpc_to_uL(mw['bulge']['rc'])
    gamma_b = mw['bulge']['gamma']
    #beta_b = mw['bulge']['beta']
    #chi_b = mw['bulge']['chi']

    #Disc: Miyamoto-Nagai
    M_d = _un.Msun_to_uM(mw['disc']['M'])
    a_d = _un.kpc_to_uL(mw['disc']['a'])
    b_d = _un.kpc_to_uL(mw['disc']['b'])

    #Dark Halo: NFW
    M_h = _un.Msun_to_uM(mw['halo']['rho']*4.0*_np.pi*mw['halo']['a']**3.0)
    #rho_h = _un.Msuninvkpc3_to_uMinvuL3(mw['halo']['rho'])
    a_h = _un.kpc_to_uL(mw['halo']['a'])
    #gamma_h = mw['halo']['gamma']
    #beta_h = mw['halo']['beta']
    q_h = mw['halo']['q']
    #--------------------------------------------
    bulge = _gl_pot.PowerSphericalPotentialwCutoff(amp=rho_b, alpha=gamma_b, rc=rc_b, r1=a_b)
    disc = _gl_pot.MiyamotoNagaiPotential(amp=M_d, a=a_d, b=b_d)

    #M_dh/q_dh to keep mass independent of q_dh
    #halo = _gl_pot.NFWPotential(amp=M_dh/q_h, a=a_dh)
    halo = _gl_pot.TriaxialNFWPotential(amp=M_h/q_h, a=a_h, b=1.0, c=q_h)
    #No C implementation
    #halo = _gl_pot.TwoPowerTriaxialPotential(amp=M_h/q_h, a=a_h, b=1.0, c=q_h, beta=3.0, alpha=gamma_h)

    potential = bulge + disc + halo
    #--------------------------------------------
    return potential


class DefineGalaxy:
    """Defines a Milky Way galaxy with a galpy potential and the circular
    velocity [kpc/Myr] at the position of the Sun [kpc]."""
    def __init__(self, R_sun, parameters_potential):
        #Galactic potential in galpy units [u.M, u.L]
        self.potential = _potential_galaxy(parameters_potential)
        #Circular velocity [kpc/Myr] at R_sun [kpc]
        self.v_lsr = _circular_velocity(R_sun, self.potential)

#-----------------------------------------------------------------------------

def _potential_spherical(sph):
    #From galactic to galpy units

    #Dark Halo: NFW
    M_h = _un.Msun_to_uM(sph['rho']*4.0*_np.pi*sph['a']**3.0)
    a_h = _un.kpc_to_uL(sph['a'])
    q_h = sph['q']

    potential = _gl_pot.TriaxialNFWPotential(amp=M_h/q_h, a=a_h, b=1.0, c=q_h)

    return potential


class DefineSpherical:
    """Defines a spherical potential with a galpy potential and the circular
    velocity [kpc/Myr] at the position of the Sun [kpc]."""
    def __init__(self, R_sun, parameters_potential):
        #Spherical potential in galpy units [u.M, u.L]
        self.potential = _potential_spherical(parameters_potential)
        #Circular velocity [kpc/Myr] at R_sun [kpc]
        self.v_lsr = _circular_velocity(R_sun, self.potential)

#-----------------------------------------------------------------------------
