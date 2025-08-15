"""Function used for the optimisation using the Inverse Time Integration
(invi) method."""

import copy as _copy
import termcolor as _tc

import numpy as _np
import scipy as _scipy
import tqdm as _tqdm

import fnc as _fnc
import invi as _invi

from ._log import *

__all__ = ["random_selection", "print_number_stars",
           "optimisation", "multiple_optimisations"]

#-----------------------------------------------------------------------------

def random_selection(x, n_sample, random_state):
    """Random selection from 'x' of 'n_sample' elements."""

    #Definition random number generator
    rng = _np.random.default_rng(random_state)
    #Number of elements
    n = len(x.T)
    #-------------------------------------------
    if n_sample > n:
        raise ValueError("n_sample <= len(x.T)")
    if n_sample == n:
        return rng.permutation(x, axis=1)
    #-------------------------------------------
    #Random initial position
    i = rng.integers(0, n)

    sel = [False]*n
    n_sel = 0

    #Generate a random boolean array
    while n_sel < n_sample:
        if sel[i] == False:
            #Random integer: 0 or 1
            if rng.integers(0, 2) == 1:
                sel[i] = True
                n_sel += 1
        i += 1
        if i == n:
            i = 0

    return rng.permutation(x, axis=1).T[sel].T


def print_number_stars(s_icrs, st_icrs, sample_icrs):
    n_s = len(s_icrs.T)
    n_st = len(st_icrs.T)
    n_sample = len(sample_icrs.T)
    _tc.cprint("Number of stars:", 'light_blue')
    print(f"Simulation = {n_s}")
    print(f"Stream     = {n_st}")
    print(f"Sample     = {n_sample} [{n_sample/n_st*100:0.3f} per cent]")

#-----------------------------------------------------------------------------

def _mu(x, sample_icrs, prm_gc, prm, options):
    #Inverse time integration in a general potential
    _, sample_alpha = _invi.inverse.integration_general_potential(x, sample_icrs, prm_gc, prm)

    #Distance from the cluster centre [rad]
    d = _np.sqrt(_np.sum(sample_alpha**2.0, axis=0))

    match options['method']:
        case 'mean':
            return _np.mean(d)
        case 'median':
            return _np.median(d)


def _mu_corrected(x, sample_icrs, prm_gc, prm, options):
    #Inverse time integration in a general potential
    sample_dgc, sample_alpha = _invi.inverse.integration_general_potential(x, sample_icrs, prm_gc, prm)

    #Stream principal axes reference frame
    #This implies that the rotation is known. This has to be evaluated in function of the potential.
    varphi = prm_gc['stream']['varphi']
    sample_AAF = _invi.coordinates.dgc_to_AAF(sample_dgc, varphi)
    sample_ALPHA = _invi.coordinates.alpha_to_ALPHA(sample_alpha, varphi)

    #Correction parameter
    mu_h = prm_gc['potential_estimation']['mu_h']

    #Correction leading arm (delta_c)
    correction = mu_h/_np.sqrt(_np.pi)*_np.array([0.0, -1.0, -1.0])

    #Frequency along the principal axis of the stream
    F1 = sample_AAF[6]

    #Leading arm correction
    sel = F1 > 0.0
    sample_ALPHA.T[sel] -= correction

    #Trailing arm correction
    sel = F1 < 0.0
    sample_ALPHA.T[sel] += correction

    #Distance from the cluster centre corrected [rad]
    d_corrected = _np.sqrt(_np.sum(sample_ALPHA**2.0, axis=0))

    match options['method']:
        case 'mean':
            return _np.mean(d_corrected)
        case 'median':
            return _np.median(d_corrected)

#-----------------------------------------------------------------------------

def _mu_within_bounds(x, sample_icrs, prm_gc, prm, options):
    if _fnc.monte_carlo.markov_chain.within_bounds(x, options['bounds']):
        if options['correction']:
            return _mu_corrected(x, sample_icrs, prm_gc, prm, options)
        return _mu(x, sample_icrs, prm_gc, prm, options)
    return _np.inf


def _loss_function(x, sample_icrs, prm_gc, prm, mu_x1, num_eval, options, log_file):
    #Number evaluation
    num_eval.n += 1

    #Evaluation function
    mu_eval = _mu_within_bounds(x, sample_icrs, prm_gc, prm, options)

    #Show information evaluation
    if options['log'] is not False:
        log_info_evaluation(x, mu_eval, mu_x1, num_eval.n, options, log_file)

    return mu_eval

#-----------------------------------------------------------------------------

def optimisation(sample_icrs, prm_gc, prm, options):
    #Number free parameters
    dim = len(options['x0'])

    #Open log file and print log header
    if options['log'] is not False:
        log_file = open(options['log'], 'w', encoding="ASCII")
        start_date, start_time = log_header(options, log_file)
    else:
        log_file = None

    #Initialise number-evaluations counter
    num_eval = NumberEvaluations()

    #Evaluation at x=1
    mu_eval_x1 = _mu_within_bounds(_np.ones(dim), sample_icrs, prm_gc, prm, options)

    #Scipy Nelder-Mead algorithm
    res = _scipy.optimize.minimize(_loss_function,
                                   x0=options['x0'],
                                   args=(sample_icrs, prm_gc, prm, mu_eval_x1, num_eval, options, log_file),
                                   method="Nelder-Mead",
                                   options={'disp': False,
                                            'maxfev': options['maxfev'],
                                            'xatol': options['xatol']})

    #Log results and close log file
    if options['log'] is not False:
        log_footer(res, dim, sample_icrs, mu_eval_x1, start_date, start_time, log_file)
        log_file.close()

    #convergence, x, mu_eval, mu_eval_x1
    return res.success, res.x, res.fun, mu_eval_x1

#-----------------------------------------------------------------------------

def multiple_optimisations(samples_icrs, prm_gc, prm, options):
    #Copy dictionary and deactivate log for the 'optimisation' function
    options_opt = _copy.deepcopy(options)
    options_opt['log'] = False

    #Open log file and print log header
    if options['log'] is not False:
        log_file = open(options['log'], 'w', encoding="ASCII")
        start_date, start_time = log_header_multiple(options, log_file)
    else:
        log_file = None

    N_opts = options['N_opts']
    success = [[]]*N_opts
    x = [[]]*N_opts
    mu_eval = _np.zeros(N_opts)
    mu_eval_x1 = _np.zeros(N_opts)

    for i in _tqdm.tqdm(range(N_opts), ncols=78, disable=not options['progress']):
        #Deactivate log for optimisation function
        success[i], x[i], mu_eval[i], mu_eval_x1[i] = optimisation(samples_icrs[i], prm_gc, prm, options_opt)

        #Log results
        if options['log'] is not False:
            log_info_optimisation(success[i], x[i], mu_eval[i], mu_eval_x1[i], i, options, log_file)

        #Save results
        res = (_np.array(x[0:i+1]).T, mu_eval[0:i+1], mu_eval_x1[0:i+1], options)
        _fnc.utils.store.save(options['results'], res, verbose=False)

    #Log results and close log file
    if options['log'] is not False:
        #Number free parameters
        dim = len(options['x0'])
        log_footer_multiple(dim, samples_icrs, start_date, start_time, log_file)
        log_file.close()

#-----------------------------------------------------------------------------
