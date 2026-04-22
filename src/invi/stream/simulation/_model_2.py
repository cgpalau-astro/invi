"""Simulation tidal stream based on Galactocentric distance."""

import numpy as np
import scipy

import invi
import invi.stream.simulation._core as ssc

__all__ = ["model_2"]

#-----------------------------------------------------------------------------

def add_structure(data):
    n = len(data['A1'])
    data['structure'] = {'all': np.array([True]*n)}
    return data

#-----------------------------------------------------------------------------

def data_from_orbit(data, prm_gc, prm, T, N=None):
    """Pericentre passages times in Myr: t in [-T, 0.0]"""
    if N is None:
        N = np.int64(T) + 1

    prm_gc['orbit']['T'] = T
    prm_gc['orbit']['N'] = N
    orb = invi.globular_cluster.orbit.bck_frw(prm_gc, prm)

    #Time from -T to 0.0 Myr
    delay = prm_gc['stream']['simulation']['delay']
    t = orb['t'] - T + delay

    #Galactocentric spherical radius
    r = orb['orbit']['FSR']['sph']['r']
    data['r_spline'] = scipy.interpolate.make_interp_spline(t, r)

    #Galactocentric cylindrical radius
    #R = orb['orbit']['FSR']['cyl']['R']
    #data['R_spline'] = scipy.interpolate.make_interp_spline(t, R)

    return data

#-----------------------------------------------------------------------------

def F1_mean_r(prm_sim, r):
    m, a, b = prm_sim['model_2']['F1_mean'].values()
    return a*r**m + b


def F1_std_r(prm_sim, r):
    m, a, b = prm_sim['model_2']['F1_std'].values()
    return a*r**m + b


def stars_stripped_r(prm_sim, r):
    m, a, b = prm_sim['model_2']['stripped_stars'].values()
    return a*r**m + b


def stripping_time_dist(prm_sim, r_spline, T, N=None):
    """Probability density function of the stripping time in function of the
    Galactocentric spherical radius estimated by a histogram.

    Note:
    -----
    1)  Histogram from the petar N-body simulation (T=1_500.0 Myr):

    import asdf
    import numpy as np
    M68 = asdf.open("../../data/M68.asdf", lazy_load=True)

    st = M68['stars']['components']['stream']
    time = M68['stars']['inv_int']['time'] #[Myr]

    hist = np.histogram(time[st], bins=100, density=False, range=[0.0, 1_500.0])
    hist_dist = scipy.stats.rv_histogram(hist, density=True)"""

    if N is None:
        N = np.int64(T)

    t = np.linspace(0.0, T, N)
    t_strip = stars_stripped_r(prm_sim, r_spline(-t))

    limit_bins = np.linspace(0.0, T, N+1)
    hist = (t_strip, limit_bins)

    hist_dist = scipy.stats.rv_histogram(hist, density=True)

    return hist_dist


def stripping_times(prm_sim, data, n_stars, rng):
    """Stripping times along orbit."""
    T = data['T']
    N = np.int64(100*T)
    hist_dist = stripping_time_dist(prm_sim, data['r_spline'], T, N)
    t_strip = hist_dist.rvs(size=n_stars, random_state=invi.misc.seed(rng))
    return t_strip


def number_stars_stripped(mass_loss, T):
    """Number stars stripped by tidal forces along the orbit."""
    n_stars = np.int64(mass_loss*T)
    return n_stars

#-----------------------------------------------------------------------------

def F_distance(data, t_strip, prm_sim, n_stars, rng):
    """Frequency along the principal axes of the stream of the stars stripped
    by tidal forces along the orbit of the cluster in function to the cluster
    Galactocentric spherical radius."""

    #Stripping radius
    r = data['r_spline'](-t_strip)
    mean = F1_mean_r(prm_sim, r) #[mrad/Myr]
    std = F1_std_r(prm_sim, r) #[mrad/Myr]

    norm = scipy.stats.norm(loc=mean, scale=std) #[mrad/Myr]
    F1 = invi.units.milli_to_unit(norm.rvs(random_state=invi.misc.seed(rng))) #[rad/Myr]

    F2 = ssc.F_2(prm_sim['F2'], n_stars, rng)

    F3 = ssc.F_3(prm_sim['F3'], n_stars, rng)

    return np.array([F1, F2, F3])


def proportional_to_distance(data, sgn, prm_sim, rng):
    """Simulate the mass loss, angular and frequency dispersion of the
    stripped stars from a globular cluster caused by tidal forces in function
    to the cluster Galactocentric spherical radius."""

    n_stars = number_stars_stripped(prm_sim['model_2']['mass_loss'], data['T'])
    t_strip = stripping_times(prm_sim, data, n_stars, rng)

    Ai = ssc.angular_dispersion(prm_sim['ang_disp'], n_stars, rng)
    Fi = F_distance(data, t_strip, prm_sim, n_stars, rng)

    A1, A2, A3 = sgn*(Fi*t_strip + Ai)
    F1, F2, F3 = sgn*Fi

    data = ssc.add_to_data(data, A1, A2, A3, F1, F2, F3)

    return data


def simulation_arm(data, arm, prm_sim, rng):

    #Selection sign for integration
    sgn = ssc.sgn_arm(arm)

    #Stars stripped by tidal forces along the orbit
    data = proportional_to_distance(data, sgn, prm_sim, rng)

    return data


def model_2(prm_gc, prm, T, random_state):
    #Parameters
    prm_sim = prm_gc['stream']['simulation']

    #Initialisation random generator
    rng1 = np.random.default_rng(random_state)
    rng2 = np.random.default_rng(random_state)

    #Initialisation data dict
    data = {'T': T}

    for key in ['A1', 'A2', 'A3', 'J1', 'J2', 'J3', 'F1', 'F2', 'F3']:
        data[key] = []

    #Determination Galactocentric sperical radius from orbit
    data = data_from_orbit(data, prm_gc, prm, T)

    #Simulation stream
    data = simulation_arm(data, 'leading', prm_sim, rng1)
    data = simulation_arm(data, 'trailing', prm_sim, rng2)

    #Definition structure dict
    data = add_structure(data)

    #Computation of actions from the eigenvalues
    data = ssc.actions_estimation(data, prm_gc['stream']['simulation']['eig'])

    #Dictionary to array conversions
    AAF = ssc.dict_to_array(data)

    return AAF, data['structure']

#-----------------------------------------------------------------------------
