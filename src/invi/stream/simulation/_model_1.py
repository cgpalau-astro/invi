"""Simulation tidal stream based on tidal shocks at pericentre passage and uniform mass loss."""

import copy
import numpy as np
import scipy

import invi
import invi.stream.simulation._core as ssc

__all__ = ["model_1"]

#-----------------------------------------------------------------------------

def add_structure(data):
    data['structure'] = {}

    n_peris = len(data['t_peris'])
    for j in range(n_peris):
        data['structure'][f"p{j+1}"] = []

    data['structure']['uni'] = []

    return data


def add_tidal_shocks_to_structure(data, n, i):
    data['structure']['uni'] += [False]*n

    n_peris = len(data['t_peris'])
    for j in range(n_peris):
        if j == i:
            data['structure'][f"p{j+1}"] += [True]*n
        else:
            data['structure'][f"p{j+1}"] += [False]*n

    return data


def add_uniform_to_structure(data, n):
    data['structure']['uni'] += [True]*n

    n_peris = len(data['t_peris'])
    for j in range(n_peris):
        data['structure'][f"p{j+1}"] += [False]*n

    return data

#-----------------------------------------------------------------------------
#Tidal shocks

def time_pericentres_r(prm_gc, prm, T, N=None):
    """Time 't' within -T <= t <= 0.0 Myr corresponding to the pericentre
    passages for a globular cluster determined with the Galactocentric
    spherical radius 'r'.

    Note
    ----
    1)  Time 't_peris' are positive."""

    if N is None:
        N = np.int64(T) + 1

    #Copy dictionary
    prm_gc_new = copy.deepcopy(prm_gc)

    prm_gc_new['orbit']['T'] = T
    prm_gc_new['orbit']['N'] = N
    orb = invi.globular_cluster.orbit.bck_frw(prm_gc_new, prm)

    #Time from -T to 0.0 Myr
    t = orb['t'] - T

    #Galactocentric cylindrical radius
    #R = orb['orbit']['FSR']['cyl']['R']

    #Galactocentric spherical radius
    r = orb['orbit']['FSR']['sph']['r']

    #Indices corresponding to the pericentres (negative to find the maximums)
    indx_peri, _ = scipy.signal.find_peaks(-r)

    return -t[indx_peri]


def time_pericentres_Ar(prm_gc, prm, T):
    """Time 't' within -T <= t <= 0.0 Myr corresponding to the pericentre
    passages for a globular cluster with initial angle 'Ar_0' [rad] and
    frequency 'Fr' [rad/Myr].

    Note
    ----
    1)  Time 't_peris' are positive."""

    #Globular cluster position in aaf
    _, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm_gc, prm)

    #Initial Ar and frequency Fr
    Ar_0 = gc_aaf[0]
    Fr = gc_aaf[6]

    #Ar corresponding to -T
    Ar_ini = invi.stream.simulation._core.t_to_Ar(-T, Ar_0, Fr)

    #Number of pericentres
    n_peris = np.abs(Ar_ini//(2.0*np.pi))

    #Ar of the pericentres
    Ar_peris = -np.flip(np.arange(0, n_peris, 1))*2.0*np.pi

    #Time pericentres
    t_peris = invi.stream.simulation._core.Ar_to_t(Ar_peris, Ar_0, Fr)

    return -t_peris

#-----------------------------------------------------------------------------

def number_stars_stripped_pericentre(par, rng):
    """Number stars stripped by tidal shocks during a pericentre passage."""

    #Uniform x in [inf, sup]
    uniform = scipy.stats.randint(low=par['peri_low'], high=par['peri_high'])
    n_stars = uniform.rvs(size=1, random_state=invi.misc.seed(rng))[0]
    return n_stars


def F_pericentre(prm_sim, n_stars, rng):
    """Frequency along the principal axes of the stream of the stars stripped
    by tidal shocks during the pericentre passages."""

    par = prm_sim['model_1']['F1_peri']
    lognorm = scipy.stats.lognorm(s=par['s'], loc=par['loc'], scale=par['scale']) #[micro_rad/Myr]
    F1 = invi.units.micro_to_unit(lognorm.rvs(size=n_stars, random_state=invi.misc.seed(rng))) #[rad/Myr]

    F2 = ssc.F_2(prm_sim['F2'], n_stars, rng)

    F3 = ssc.F_3(prm_sim['F3'], n_stars, rng)

    return np.array([F1, F2, F3])


def tidal_shocks(data, sgn, prm_sim, rng):
    """Simulate the mass loss, angular and frequency dispersion of the
    stripped stars from a globular cluster caused by a tidal shock
    during a pericentre passage."""

    for i, t_peri in enumerate(data['t_peris']):
        n_stars = number_stars_stripped_pericentre(prm_sim['model_1']['mass_loss'], rng)

        Ai = ssc.angular_dispersion(prm_sim['ang_disp'], n_stars, rng)
        Fi = F_pericentre(prm_sim, n_stars, rng)

        A1, A2, A3 = sgn*(Fi*t_peri + Ai)
        F1, F2, F3 = sgn*Fi

        data = ssc.add_to_data(data, A1, A2, A3, F1, F2, F3)

        data = add_tidal_shocks_to_structure(data, n_stars, i)

    return data

#-----------------------------------------------------------------------------
#Intervals mass loss

def normalised_int_fraction(frac_arr, n):
    """n*frac to np.int64 such that its sum is equal to n:

    Note
    ----
    1) The residual is divided in units and added to the first elements of the list:

    Example
    -------
    1)  frac_arr = np.array([1/3, 1/3, 1/3])
        n = 2_333
        norm_frac = normalised_int_fraction(frac_arr, n)
        >>> array([778, 778, 777])"""

    n_int = np.int64(n*frac_arr)
    n_sum = np.sum(n_int)

    for i in range(n%n_sum):
        n_int[i] += 1

    return n_int


def intervals_stripping_time(n_stars, T, t_peris, t_intv, rng):
    """Stripping times during intervals along orbit.

    Note
    ----
    1)  T, t_peris > 0.0 [Myr]"""

    #Number pericentre passages
    n_peris = len(t_peris)

    #Define limits intervals (T and t_peris > 0.0)
    limits_intv = [-T]
    for t_peri in -t_peris:
        limits_intv.append(t_peri - t_intv)
        limits_intv.append(t_peri + t_intv)
    limits_intv.append(0.0)

    #Number intervals when stars are stripped
    n_intv = n_peris + 1

    #Time intervals when stars are stripped
    time_intv = -np.array([limits_intv[i] - limits_intv[i+1] for i in range(0, n_intv*2, 2)])

    #Eliminate negative intervals (overlaping or out of boundaries peaks)
    time_intv[time_intv<0] = 0.0

    #Fraction stars stripped each interval
    frac = time_intv/np.sum(time_intv)

    #Number stars stripped each interval
    n_stars_intv = normalised_int_fraction(frac, n_stars)

    #Uniform random generation of stripping times within intervals
    t = []
    for i in range(n_intv):
        uniform = scipy.stats.uniform(loc=limits_intv[i*2], scale=time_intv[i])
        t += list(uniform.rvs(size=n_stars_intv[i], random_state=invi.misc.seed(rng)))

    t = -np.array(t)

    return t


def intervals_mass_loss(data, sgn, prm_sim, rng):
    """Simulate the mass loss, angular and frequency dispersion of the
    stripped stars from a globular cluster caused by tidal forces between
    pericentre passages."""

    n_stars = number_stars_stripped_uniform(prm_sim['model_1']['mass_loss']['uni'], data['T'])

    t_intv = 50.0 #[Myr]
    t_intvs = intervals_stripping_time(n_stars, data['T'], data['t_peris'], t_intv, rng)

    Ai = ssc.angular_dispersion(prm_sim['ang_disp'], n_stars, rng)
    Fi = F_uniform(prm_sim, n_stars, rng)

    A1, A2, A3 = sgn*(Fi*t_intvs + Ai)
    F1, F2, F3 = sgn*Fi

    data = ssc.add_to_data(data, A1, A2, A3, F1, F2, F3)

    data = add_uniform_to_structure(data, n_stars)

    return data

#-----------------------------------------------------------------------------
#Uniform mass loss

def uniform_stripping_time(n_stars, T, rng):
    """Stripping times along orbit."""
    #Uniform x in [-T, 0.0]
    uniform = scipy.stats.uniform(loc=-T, scale=T)
    t = np.abs(uniform.rvs(size=n_stars, random_state=invi.misc.seed(rng)))
    return t


def number_stars_stripped_uniform(mass_loss, T):
    """Number stars stripped by tidal forces uniformly along the orbit."""
    n_stars = np.int64(mass_loss*T)
    return n_stars


def F_uniform(prm_sim, n_stars, rng):
    """Frequency along the principal axes of the stream of the stars stripped
    by tidal forces along the orbit of the cluster."""

    par = prm_sim['model_1']['F1_uni']
    gennorm = scipy.stats.gennorm(beta=par['beta'], loc=par['loc'], scale=par['scale']) #[micro_rad/Myr]
    F1 =  invi.units.micro_to_unit(gennorm.rvs(size=n_stars, random_state=invi.misc.seed(rng))) #[rad/Myr]

    F2 = ssc.F_2(prm_sim['F2'], n_stars, rng)

    F3 = ssc.F_3(prm_sim['F3'], n_stars, rng)

    return np.array([F1, F2, F3])


def uniform_mass_loss(data, sgn, prm_sim, rng):
    """Simulate the mass loss, angular and frequency dispersion of the
    stripped stars from a globular cluster caused by tidal forces between
    pericentre passages."""

    n_stars = number_stars_stripped_uniform(prm_sim['model_1']['mass_loss']['uni'], data['T'])

    t_uni = uniform_stripping_time(n_stars, data['T'], rng)

    Ai = ssc.angular_dispersion(prm_sim['ang_disp'], n_stars, rng)
    Fi = F_uniform(prm_sim, n_stars, rng)

    A1, A2, A3 = sgn*(Fi*t_uni + Ai)
    F1, F2, F3 = sgn*Fi

    data = ssc.add_to_data(data, A1, A2, A3, F1, F2, F3)

    data = add_uniform_to_structure(data, n_stars)

    return data

#-----------------------------------------------------------------------------

def simulation_arm(data, arm, prm_sim, rng):

    #Selection sign for integration
    sgn = ssc.sgn_arm(arm)

    #Stars stripped by tidal shocks during the pericentre passages
    data = tidal_shocks(data, sgn, prm_sim, rng)

    #Stars stripped by tidal forces along the orbit
    data = uniform_mass_loss(data, sgn, prm_sim, rng)
    #data = intervals_mass_loss(data, sgn, prm_sim, rng)

    return data


def model_1(prm_gc, prm, T, random_state):
    #Parameters
    prm_sim = prm_gc['stream']['simulation']

    #Initialisation random generator
    rng1 = np.random.default_rng(random_state)
    rng2 = np.random.default_rng(random_state)

    #Initialisation data dict
    data = {'T': T}

    for key in ['A1', 'A2', 'A3', 'J1', 'J2', 'J3', 'F1', 'F2', 'F3']:
        data[key] = []

    #Determination time pericentres passages from Galactocentric spherical radius
    #t_peris = time_pericentres_r(prm_gc, prm, T, N=np.int64(T*10)+1)

    #Determination time pericentres passages from angle Ar of the globular cluster
    t_peris = time_pericentres_Ar(prm_gc, prm, T)

    #Delay between the maximum mass loss and the pericentre passage
    data['t_peris'] = t_peris - prm_sim['delay']

    #Definition structure dict
    data = add_structure(data)

    #Simulation stream
    data = simulation_arm(data, 'leading', prm_sim, rng1)
    data = simulation_arm(data, 'trailing', prm_sim, rng2)

    #Computation of actions from the eigenvalues
    data = ssc.actions_estimation(data, prm_gc['stream']['simulation']['eig'])

    #Dictionary to array conversions
    AAF = ssc.dict_to_array(data)
    structure = ssc.to_array(data['structure'])

    return AAF, structure

#-----------------------------------------------------------------------------
