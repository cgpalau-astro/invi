"""Phase-space models for globular clusters."""

import numpy as _np
import tqdm as _tqdm

import fnc as _fnc
_agama = _fnc.utils.lazy.Import("agama")

import invi as _invi
import invi.units as _un

#Set agama in galpy units
_agama.setUnits(length=_un.u.L, velocity=_un.u.V, mass=_un.u.M)

__all__ = ["Plummer", "King"]

#-----------------------------------------------------------------------------

class Plummer:
    """Plummer distribution."""
    class rvs:
        """Random variable sample."""
        @staticmethod
        def analytical(mass, a, size, random_state=None, progress=False):
            """Generate phase-space random sample following a Plummer model.

            Parameters
            ----------
            mass : float
                Mass [M_sun]
            a : float
                Scale length [pc]
            size : int
                Number of stars
            random_state : int
                Seed
            progress : bool
                Print progress bar

            Returns
            -------
            np.array
                Phase-space coordinates of the stars in Galactic units.
                {x, y, z, vx, vy, vz} [kpc, kpc/Myr]"""
            #-------------------------------------------------------
            G_const = 4.4987E-12 #[kpc**3/(Myr**2*M_sun)]

            def rand_angle(n):
                costheta = _np.random.uniform(-1.0, 1.0, n)
                theta = _np.arccos(costheta)
                phi = _np.random.uniform(0.0, 2.0*_np.pi, n)
                return theta, phi

            def sph_to_car(r, theta, phi):
                x = r * _np.sin(theta) * _np.cos(phi)
                y = r * _np.sin(theta) * _np.sin(phi)
                z = r * _np.cos(theta)
                return x, y, z

            def p_dist(q):
                return 512.0/(7.0*_np.pi)*q**2.0 * (1.0-q**2)**3.5
            #-------------------------------------------------------
            #Definition seed
            _np.random.seed(random_state)

            #All in Galactic units
            a = _un.pc_to_kpc(a) #[kpc]

            #Position distribution
            u = _np.random.uniform(0.0, 1.0, size)
            r = a/_np.sqrt( u**(-2.0/3.0) - 1.0 )

            theta, phi = rand_angle(size)
            x, y, z = sph_to_car(r, theta, phi) #[kpc]

            #Velocity distribution
            c = 50176.0*_np.sqrt(7.0)/(19683.0*_np.pi)
            vr = _np.zeros(size)
            for i in _tqdm.tqdm(range(size), disable=not progress, ncols=78):
                q = 0.0
                u = 0.1
                while u > p_dist(q):
                    q = _np.random.uniform(0.0, 1.0)
                    u = _np.random.uniform(0.0, c)
                vr[i] = q * _np.sqrt(2.0*mass*G_const) * (a**2.0 + r[i]**2.0)**(-0.25)

            theta, phi = rand_angle(size)
            px, py, pz = sph_to_car(vr, theta, phi) #[kpc/Myr]

            sample = _np.array([x, y, z, px, py, pz])

            return sample

        @staticmethod
        def agama(mass, a, size):
            """Generate phase-space random sample following a Plummer model using an
            AGAMA potential.

            Parameters
            ----------
            mass : float
                Mass [M_sun]
            a : float
                Scale length [pc]
            size : int
                Number stars

            Returns
            -------
            np.array
                Phase-space coordinates of the stars in Galactic units.
                {x, y, z, vx, vy, vz} [kpc, kpc/Myr]"""
            #-------------------------------------------------------------------------
            #To Galpy units
            mass = _un.Msun_to_uM(mass)
            a = _un.pc_to_uL(a)

            plummer_potential = _agama.Potential(type='Plummer', mass=mass, scaleRadius=a)
            sample = _invi.agama.distribution_function.rvs(plummer_potential, size) #[kpc, kpc/Myr]

            return sample

#-----------------------------------------------------------------------------

class King:
    """Generalised spherical lowered isothermal model.

    Note
    ----
    1)  Globular cluster models database: arXiv:1901.08072

    2)  W0     : Dimensionless central potential
        g      : Truncation parameter: 0=Woolley, 1=Michie-King, 2=Wilson
        r_truncation : Truncation radius
        r_core : Core radius
        r_half : Half mass radius

    1)  In arXiv:1901.08072 it is given: W0, g, r_half, r_truncation. Agama
        requires r_core. This parameter has to be calculated numerically
        using the astro_limepy package. This functionality is provided in
        astro_limepy.utils.limepy_model_parameters."""

    class rvs:
        """Random variable sample."""
        @staticmethod
        def limepy(mass, r_core, W0, g, size, random_state=123):
            """Generate phase-space random sample following a King model using astro_limepy.

            Note
            ----
            1)  This method produces irregularities in the generated sample.
                It is better to use Agama.

            Parameters
            ----------
            mass : float
                Mass [M_sun]
            r_core : float
                Core radius [pc]
            W0 : float
                W0 parameter [-]
            g : float
                Truncation parameter [-]
            size : int
                Number stars
            random_state : int
                Seed

            Returns
            -------
            np.array
                Phase-space coordinates of the stars in Galactic units.
                {x, y, z, vx, vy, vz} [kpc, kpc/Myr]"""
            #-------------------------------------------------------------------------
            model = _invi.astro_limepy.limepy(W0, g, M=mass, r0=r_core, verbose=False)
            x = _invi.astro_limepy.sample(model, seed=random_state, verbose=False, N=size) #[pc, km/s]

            #Limepy to Astronomic units
            sample = _np.array([_un.pc_to_kpc(x.x), _un.pc_to_kpc(x.y), _un.pc_to_kpc(x.z),
                                x.vx, x.vy, x.vz]) #[kpc, km/s]
            #Astronomic to Galactic units
            sample = _un.astronomic_to_galactic(sample)

            return sample

        @staticmethod
        def agama(mass, r_core, W0, g, size):
            """Generate phase-space random sample following a King model using an AGAMA
            potential.

            Parameters
            ----------
            mass : float
                Mass [M_sun]
            r_core : float
                Core radius [pc]
            W0 : float
                W0 parameter [-]
            g : float
                Truncation parameter [-]
            size : int
                Number stars

            Returns
            -------
            np.array
                Phase-space coordinates of the stars in Galactic units.
                {x, y, z, vx, vy, vz} [kpc, kpc/Myr]"""
            #-------------------------------------------------------------------------
            mass = _un.Msun_to_uM(mass)
            r_core = _un.pc_to_uL(r_core)

            potential_king = _agama.Potential(type="King", mass=mass, scaleRadius=r_core, W0=W0, trunc=g)

            nbody = _invi.agama.distribution_function.rvs(potential_king, size) #[kpc, kpc/Myr]

            return nbody

#-----------------------------------------------------------------------------
