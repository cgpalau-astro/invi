"""Self-consistent random sample following a distribution given by an agama
potential."""

#import agama as _agama
from fnc.utils import lazy as _lazy
_agama = _lazy.Import("agama")

import invi.units as _un

#Set Agama in Galpy units
_agama.setUnits(mass=_un.u.M, length=_un.u.L, velocity=_un.u.V)

__all__ = ["rvs"]

#-----------------------------------------------------------------------------

def rvs(agama_potential, size, number_iterations=10, verbose=False):
    """Generate a phase-space random sample following an agama potential.

    Note
    ----
    1)  There is no way to specify the seed for agama samples.

    Parameters
    ----------
    agama_potential : agama.potential
        Agama potential
    size : int
        Number of stars sample
    number_iterations : int
        Number of iterations
    verbose : bool
        Print agama messages

    Returns
    -------
    np.array
        Phase-space random sample in galactic units [kpc, kpc/Myr]
        shape(nbody) = (6, size)"""
    #-------------------------------------------------------------
    #Definition of the distribution function defining the model
    df = _agama.DistributionFunction(type='QuasiSpherical', potential=agama_potential)

    #Define the self-consistent model consisting of a single component
    #sizeRadialSph: Number stored points
    params = {'rminSph':1.0E-8, 'rmaxSph':10.0, 'sizeRadialSph':100, 'lmaxAngularSph':0}
    #comp = _agama.Component(df=df, density=agama_potential.density, disklike=False, **params)
    comp = _agama.Component(df=df, density=agama_potential, disklike=False, **params)

    #Define self consistent model with one component
    scm = _agama.SelfConsistentModel(**params, verbose=verbose)
    scm.components = [comp]

    #Iterations
    for _ in range(number_iterations):
        scm.iterate()

    #Generate random sample
    sample = _agama.GalaxyModel(scm.potential, df).sample(size)

    #Convert from galpy to galactic units
    sample = _un.galpy_to_galactic(sample[0].T)

    return sample

#-----------------------------------------------------------------------------
