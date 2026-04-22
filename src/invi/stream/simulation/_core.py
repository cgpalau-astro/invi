"""Core functions for simulation module."""

import numpy as np
import scipy

import fnc
import invi
import invi.stream.simulation._model_3 as model_3

__all__ = ["classify", "gc_r_truncation", "t_to_Ar", "Ar_to_t",
           "fit_parameters", "limit"]

#-----------------------------------------------------------------------------

def classify(prm_gc, sim_AAF):
    """Classification of simulated stream stars equal to the classification of
    the N-body stars."""

    n = len(sim_AAF[0])
    true = np.array([True]*n)
    false = np.array([False]*n)

    components = {}
    components['gc'] = false
    components['escapees'] = false
    components['stream'] = true
    components['leading'] = components['stream'] & (sim_AAF[6] > 0.0) #s_AAF[0] = A1, s_AAF[6] = A1
    components['trailing'] = components['stream'] & (sim_AAF[6] < 0.0)

    #Time integration
    time = invi.inverse.integration_time(sim_AAF)

    #Internal streams
    components['internal'] = invi.stars.components._internal_streams(prm_gc, time, components)

    return components


def gc_r_truncation(prm_gc, s_fsr, gc_fsr):
    """Stars belonging to the globular cluster defined by the truncation
       radius [kpc]."""

    #Relative position to globular cluster [kpc]
    x = s_fsr[0] - gc_fsr[0]
    y = s_fsr[1] - gc_fsr[1]
    z = s_fsr[2] - gc_fsr[2]

    d = _np.sqrt(x**2.0 + y**2.0 + z**2.0)*1_000.0 #[pc]
    gc = d < prm_gc['king']['r_truncation']

    return gc

#-----------------------------------------------------------------------------

def add_to_data(data, A1, A2, A3, F1, F2, F3):
    data['A1'] = np.append(data['A1'], A1)
    data['A2'] = np.append(data['A2'], A2)
    data['A3'] = np.append(data['A3'], A3)

    data['F1'] = np.append(data['F1'], F1)
    data['F2'] = np.append(data['F2'], F2)
    data['F3'] = np.append(data['F3'], F3)
    return data

#-----------------------------------------------------------------------------

def dict_to_array(data):
    AAF = []
    for key in ['A1', 'A2', 'A3', 'J1', 'J2', 'J3', 'F1', 'F2', 'F3']:
        AAF.append(np.array(data[key]))
    return np.array(AAF)


def to_array(structure):
    structure_arr = {}
    for key in structure.keys():
        structure_arr[key] = np.array(structure[key])
    return structure_arr

#-----------------------------------------------------------------------------

def actions_estimation(data, eig):
    """Computation of actions from the frequencies and eigenvalues."""
    data['J1'] = [x/eig[0] for x in data['F1']]
    data['J2'] = [x/eig[1] for x in data['F2']]
    data['J3'] = [x/eig[2] for x in data['F3']]
    return data

#-----------------------------------------------------------------------------

def F_2(par, n_stars, rng):
    norm = scipy.stats.norm(loc=par['loc'], scale=par['scale']) #[micro_rad/Myr]
    F2 = invi.units.micro_to_unit(norm.rvs(size=n_stars, random_state=invi.misc.seed(rng))) #[rad/Myr]
    return F2


def F_3(par, n_stars, rng):
    norm = scipy.stats.norm(loc=par['loc'], scale=par['scale']) #[micro_rad/Myr]
    F3 = invi.units.micro_to_unit(norm.rvs(size=n_stars, random_state=invi.misc.seed(rng))) #[rad/Myr]
    return F3

#-----------------------------------------------------------------------------

def angular_dispersion(par, n_stars, rng):
    """Initial angular dispersion in rad"""

    multi_norm = scipy.stats.multivariate_normal(mean=par['mean'], cov=par['cov']) #[mrad]
    Ai = invi.units.milli_to_unit(multi_norm.rvs(size=n_stars, random_state=invi.misc.seed(rng)).T) #[rad]

    return Ai

#-----------------------------------------------------------------------------

def sgn_arm(arm):
    """Sign integration corresponding to each arm of the stream."""
    match arm:
        case 'leading':
            sgn = 1.0
        case 'trailing':
            sgn = -1.0
        case _:
            raise ValueError("arm = {'leading', 'trailing'}")
    return sgn

#-----------------------------------------------------------------------------

def t_to_Ar(t, Ar_0, Fr):
    """From time 't' [Myr] to angle 'Ar' [rad] for a particle with initial
    angle 'Ar_0' [rad] and frequency 'Fr' [rad/Myr]"""
    return Ar_0 + Fr*t


def Ar_to_t(Ar, Ar_0, Fr):
    """From angle 'Ar' [rad] to time 't' [Myr] for a particle with initial
    angle 'Ar_0' [rad] and frequency 'Fr' [rad/Myr]"""
    return (Ar - Ar_0)/Fr

#-----------------------------------------------------------------------------

def fit_parameters(x, y, x0):
    """Fit parameters to a double exponential model."""
    kwargs = {'method': 'trf', 'nan_policy': 'omit', 'maxfev': 10_000,
              'ftol':1E-12, 'xtol':1E-12, 'gtol':1E-12}

    def double_exp(Ar, A, C, tau_1, tau_2):
        coef = np.array([A, C, tau_1, tau_2])
        return model_3.double_exp(Ar, coef)

    coef, _ = scipy.optimize.curve_fit(double_exp, x, y, p0=x0, **kwargs)

    return coef


def limit(coef):
    """Point where the two exponentials of the double exponential model are
    equal."""
    tau_1 = coef[2]
    tau_2 = coef[3]
    return 2.0*np.pi*tau_1/(tau_1 + tau_2)

#-----------------------------------------------------------------------------
