"""Power-law mass function.

Example 1
---------
import numpy as np
import fnc

alpha = -1.27
a = 0.2
b = 0.96
loc = 0.0
size = 432_384
random_state = 123

x = np.linspace(a, b, 1_000)
pdf = fnc.stats.powerlaw.pdf(x, loc=loc, alpha=alpha, a=a, b=b)
sample = fnc.stats.powerlaw.rvs(loc=loc, alpha=alpha, a=a, b=b, size=size, random_state=random_state)

norm_mean = np.mean(sample) / fnc.stats.powerlaw.mean(loc=loc, alpha=alpha, a=a, b=b)
norm_std = np.std(sample) / fnc.stats.powerlaw.std(loc=loc, alpha=alpha, a=a, b=b)

print(f"Normalised mean = {norm_mean:0.5f}")
print(f"Normalised std  = {norm_std:0.5f}")

fig, ax = fnc.plot.figure()
ax.plot(x, pdf)
h = ax.hist(sample, bins=100, density=True, range=(a, b), histtype="step")
#ax.set_xlim(a, b)
ax.set_xlabel("x")
ax.set_ylabel("pdf")

Example 2
---------
import numpy as np
import fnc
import invi.globular_cluster.mass_function as mf

mass = 125_000
alpha = -1.27
a = 0.2
b = 0.96
random_state = 1234

def mode_rounding(rounding):
    number_stars = mf.number_stars(mass=mass, alpha=alpha, a=a, b=b, random_state=random_state, rounding=rounding, factor=0.01)
    sample = fnc.stats.powerlaw.rvs(alpha=alpha, a=a, b=b, size=number_stars, random_state=random_state)
    total_mass = np.sum(sample)
    print(f"Method = {rounding}")
    print(f"Number stars = {number_stars:_}")
    print(f"Total mass   = {total_mass:0.5f}")
    print()

mode_rounding('superior')
mode_rounding('inferior')"""

import numpy as _np
import fnc as _fnc

__all__ = ["number_stars"]

#-----------------------------------------------------------------------------

def number_stars(mass, alpha, a, b, random_state, rounding='superior', factor=0.05):
    """Number of stars with mass following 'power_law.rvs(alpha, a, b)' for a
    given 'random_state' such that their total mass is approximately equal to
    'mass'.

    Note
    ----
    1)  Small 'factor' for faster evaluation.

    Parameters
    ----------
    mass : float
        Total mass
    alpha : float
        Slope power-law
    a : float
        Inferior limit power-law
    b : float
        Superior limit power-law
    random_state : int
        Seed
    rounding : {'inferior', 'superior'}
        Inferior or superior rounding
    factor: float
        Determines the size of the random sample used to compute the number of
        stars (0.0 < factor)

    Returns
    -------
    numpy.int64 | None
        Number of stars"""
    #-----------------------------------------------------------
    if factor <= 0.0:
        raise ValueError("'factor' out of bounds: factor > 0.0")

    if rounding not in ('inferior', 'superior'):
        raise ValueError("rounding = {inferior, superior}")
    #-----------------------------------------------------------
    #Determine size sample
    average_mass_star = _fnc.stats.powerlaw.mean(loc=0.0, alpha=alpha, a=a, b=b)
    fraction = 1.0 + factor
    size = _np.int64(fraction*mass/average_mass_star)

    #Generate random sample
    sample = _fnc.stats.powerlaw.rvs(loc=0.0, alpha=alpha, a=a, b=b, size=size, random_state=random_state)

    if _np.sum(sample) < mass:
        raise ValueError(f"It is necessary a factor > {factor}")

    #Determine number of stars
    n_stars = _np.argmin( _np.abs( _np.cumsum(sample) - mass ) ) + 1

    #Apply rounding to n_stars
    if rounding == 'superior':
        if _np.sum(sample[0:n_stars]) < mass:
            n_stars += 1

    if rounding == 'inferior':
        if _np.sum(sample[0:n_stars]) > mass:
            n_stars -= 1

    return n_stars

#-----------------------------------------------------------------------------
