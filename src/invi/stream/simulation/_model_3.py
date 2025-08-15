"""Simulation tidal stream based on Ar.

Note
----
1)  The eigenvalues are precomputed and stored as parameters. The aaf of the
    cluster is evaluated in function of the potential and initial conditions.

2)  The frequencies and stripping times are given in function of the radial
    angle Ar. This angle is corrected by the delay such that the peaks
    correspond to Ar=[..., 2pi, pi, 0, pi, 2pi, ...]. It is therefore
    necessary to improve the notation to avoid confusion."""

import numpy as np
import scipy

import fnc
import invi
import invi.stream.simulation._core as ssc

__all__ = ["model_3"]

#-----------------------------------------------------------------------------

def add_structure(data):
    n = len(data['A1'])
    data['structure'] = {'all': np.array([True]*n)}
    return data

#-----------------------------------------------------------------------------

def wrap(x):
    """x rem 2*pi"""
    return x % (2.0*np.pi)


def centre(x):
    """From x in [0, 2*pi] to [-pi, pi]"""
    return np.where(x<=np.pi, x, x - 2.0*np.pi)


def desp(x, limit):
    return np.where(x<=limit, x, x - 2.0*np.pi)


def double_exp(Ar, coef):
    """Double exponential model of the peaks in Ar [rad].

    Note
    ----
    1)  A : Scale high [unit]
        C : Base line [unit]
        tau_1: Scale length decay [rad]
        tau_2: Scale length ascend [rad]

    2)  Ar in [0.0, 2.0*pi] rad but double_exp(0.0) != double_exp(2.0*pi)"""

    A, C, tau_1, tau_2 = coef

    #It is necessary to conver to array for the numerical integration with 'quad'
    dexp = np.asarray( A*( np.exp(-Ar/tau_1) + np.exp( (Ar-2.0*np.pi)/tau_2 ) ) + C )

    out = np.logical_not(fnc.numeric.within_equal(Ar, 0.0, 2.0*np.pi))
    dexp[out] = np.nan

    return dexp


def double_exp_wrap(Ar, coef):
    return double_exp(wrap(Ar), coef)

#-----------------------------------------------------------------------------

def globular_cluster_radial_coordinates(data, prm_gc, prm):

    #Globular cluster aaf at the current position
    _, gc_aaf = invi.globular_cluster.orbit.ic_FSR_aaf(prm_gc, prm)

    #Current radial angle and frequency
    data['gc_Ar_0'] = gc_aaf[0] #[rad]
    data['gc_Fr'] = gc_aaf[6] #[rad/Myr]

    #Globular cluster initial Ar [rad] (at -T)
    data['gc_Ar_T'] = ssc.t_to_Ar(-data['T'], data['gc_Ar_0'], data['gc_Fr'])

    return data

#-----------------------------------------------------------------------------

def number_stripped_stars(mass_loss, T):
    """Number stars stripped by tidal forces along the orbit."""
    n_stars = np.int64(mass_loss*T)
    return n_stars


class NormalisationConstant():
    """Compute normalisation constant of the 'pdf_double_exp_wrap' function."""

    @staticmethod
    def numeric(Ar_T, Ar_0, coef, **kwargs):
        return scipy.integrate.quad(double_exp_wrap, a=Ar_T, b=Ar_0, args=(coef,), **kwargs)[0]

    @staticmethod
    def exact(Ar_T, Ar_0, coef):

        def area_double_exp(Ar_inf, Ar_sup, coef):
            A, C, tau_1, tau_2 = coef
            h = np.exp(-Ar_inf/tau_1) - np.exp(-Ar_sup/tau_1)
            g = np.exp( Ar_sup/tau_2) - np.exp( Ar_inf/tau_2)
            return C*(Ar_sup - Ar_inf) + A*h*tau_1 + A*g*tau_2*np.exp(-2.0*np.pi/tau_2)

        #0 <= A_inf < Ar_sup
        if Ar_T >= 0.0:
            return area_double_exp(Ar_T, Ar_0, coef)

        nc_positive = area_double_exp(0.0, Ar_0, coef)

        nc_period = area_double_exp(0.0, 2.0*np.pi, coef)
        n_periods, Ar_reminder = divmod(np.abs(Ar_T), 2.0*np.pi)

        nc_reminder = area_double_exp(2.0*np.pi-Ar_reminder, 2.0*np.pi, coef)

        return nc_reminder + nc_period*n_periods + nc_positive


def pdf_double_exp_wrap(Ar, data, coef):
    """ Note
        ----
        1)  The evaluation of the normalisation constat is avoidable because
            it is computed numerically when the CDF is normalised."""
    return double_exp_wrap(Ar, coef) / NormalisationConstant.exact(data['gc_Ar_T'], data['gc_Ar_0'], coef)


def stripping_Ar(data, coef, n_stars, rng, N=None):
    """Random sample of stripping Ar [rad].

    Note
    ----
    1)  This distribution is calibrated assuming that the peaks of the
        distribution of stripping points are located at
        Ar=[..., 2pi, pi, 0, pi, 2pi, ...]. There is a delay between Ar and
        the peaks. This delay is corrected when computing the stripping time
        (t_strip)."""

    if N is None:
        N = np.int64(data['T'])

    dist_spline = fnc.stats.DistSpline(pdf_double_exp_wrap,
                                       x_inf=data['gc_Ar_T'],
                                       x_sup=data['gc_Ar_0'],
                                       n_points=N)

    #Generate random sample
    random_state = invi.misc.seed(rng)
    Ar_strip = dist_spline.rvs(n_stars, random_state, data, coef)

    return Ar_strip

#-----------------------------------------------------------------------------

def F_Ar(Ar_strip, prm_sim, n_stars, rng):
    """Frequency along the principal axes of the stream of the stars stripped
    by tidal forces along the orbit of the cluster in function to the cluster
    radial angle Ar [rad].

    Note
    ----
    1)  These distributions are calibrated assuming that the peaks of the
        distribution of stripping points are located at
        Ar=[..., 2pi, pi, 0, pi, 2pi, ...]. There is a delay between Ar and
        the peaks. This delay is corrected when computing the stripping time
        (t_strip)."""

    mean = double_exp_wrap(Ar_strip, prm_sim['model_3']['F1_mean_coef']) #[mrad/Myr]
    std = double_exp_wrap(Ar_strip, prm_sim['model_3']['F1_std_coef']) #[mrad/Myr]
    norm = scipy.stats.norm(loc=mean, scale=std) #[mrad/Myr]

    F1 = invi.units.milli_to_unit(norm.rvs(random_state=invi.misc.seed(rng))) #[rad/Myr]

    F2 = ssc.F_2(prm_sim['F2'], n_stars, rng)

    F3 = ssc.F_3(prm_sim['F3'], n_stars, rng)

    return np.array([F1, F2, F3])


def proportional_to_Ar(data, kappa, prm_sim, rng):
    """Simulate the mass loss, angular and frequency dispersion of the
    stripped stars from a globular cluster caused by tidal forces in function
    to the cluster radial angle Ar [rad]."""

    n_stars = number_stripped_stars(prm_sim['model_3']['mass_loss'], data['T'])
    Ar_strip = stripping_Ar(data, prm_sim['model_3']['stripped_stars_coef'], n_stars, rng)
    Ai = ssc.angular_dispersion(prm_sim['ang_disp'], n_stars, rng)
    Fi = F_Ar(Ar_strip, prm_sim, n_stars, rng)

    #Stripping time [Myr] including delay correction
    t_strip = -ssc.Ar_to_t(Ar_strip, data['gc_Ar_0'], data['gc_Fr']) - prm_sim['delay']

    A1, A2, A3 = kappa*(Fi*t_strip + Ai)
    F1, F2, F3 = kappa*Fi

    data = ssc.add_to_data(data, A1, A2, A3, F1, F2, F3)

    return data


def simulation_arm(data, arm, prm_sim, rng):

    #Selection sign for integration
    kappa = ssc.sgn_arm(arm)

    #Stars stripped by tidal forces along the orbit
    data = proportional_to_Ar(data, kappa, prm_sim, rng)

    return data


def model_3(prm_gc, prm, T, random_state):
    #Parameters
    prm_sim = prm_gc['stream']['simulation']

    #Initialisation random generator
    rng1 = np.random.default_rng(random_state)
    #rng2 = np.random.default_rng(random_state)

    #Initialisation data dict
    data = {'T': T}

    for key in ['A1', 'A2', 'A3', 'J1', 'J2', 'J3', 'F1', 'F2', 'F3']:
        data[key] = []

    #Determination Ar and Fr of the globular cluster
    data = globular_cluster_radial_coordinates(data, prm_gc, prm)

    #Simulation stream
    data = simulation_arm(data, 'leading', prm_sim, rng1)
    data = simulation_arm(data, 'trailing', prm_sim, rng1)

    #Definition structure dict. It is empty but included for symmetry with model 1.
    data = add_structure(data)

    #Computation of actions from the eigenvalues
    data = ssc.actions_estimation(data, prm_gc['stream']['simulation']['eig'])

    #Dictionary to array conversions
    AAF = ssc.dict_to_array(data)

    return AAF, data['structure']

#-----------------------------------------------------------------------------
