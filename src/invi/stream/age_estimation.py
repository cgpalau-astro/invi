"""Functions to estimate the age of an stellar stream."""

import numpy as _np
import scipy as _scipy

import fnc as _fnc
import invi as _invi

_tqdm = _fnc.utils.lazy.Import("tqdm")

__all__ = ["ages",
           "load_surface_density_data",
           "DistributionKDE", "DistributionHist",
           "posterior"]

#-----------------------------------------------------------------------------

def ages(T_ini, T_end, intv, verbose=False):
    #T in ['T_ini', 'T_end'] with interval 'intv' in Myr.
    T = _np.arange(T_ini, T_end+intv, intv)
    if verbose:
        print(T)
    return T

#-----------------------------------------------------------------------------

def load_surface_density_data(name_folder, T):
    """Load simulated surface density of age 'T' in Myr. Each file includes n
    samples."""
    res = _fnc.utils.store.load(f'../../data/misc/surface_density/{name_folder}/res.opy')

    n_samples = len(res)
    n_stars = []
    A1_sim = []

    for data in res:
        A1_orb_est = data['A1_orb_est']

        #Stripping time of each star
        sim_stripping_time = -_invi.inverse.integration_time(data['sim_AAF']) #[Myr]

        #sel_sim and stripping time cuts
        sel = data['sel'] & (sim_stripping_time > -T)

        #Number of stars of each sample
        n_stars.append(len(A1_orb_est[sel]))

        #A1 of each star
        A1_sim += list(A1_orb_est[sel])


    A1_sim = _np.array(A1_sim)

    mean_stars = _np.mean(n_stars)
    std_stars = _np.std(n_stars)

    return n_samples, A1_sim, mean_stars, std_stars

#-----------------------------------------------------------------------------

@_np.vectorize
def _pdf(x, self):
    if (x < self.range[0]) | (self.range[1] < x):
        return 0.0
    return self.kde(x)/self.norm_const


class DistributionKDE:
    def __init__(self, A1_data, bw, rng):
        self.range = rng

        kde = _scipy.stats.gaussian_kde(A1_data, bw_method=bw)
        self.kde = kde

        if bw == 'silverman':
            self.bw = kde.silverman_factor()
        else:
            self.bw = bw

        self.norm_const, self.norm_const_err = _scipy.integrate.quad(kde, a=rng[0], b=rng[1])


    def likelihood(self, x):
        return _np.prod(_pdf(x, self))


    def pdf(self, x):
        return _pdf(x, self)

#-----------------------------------------------------------------------------

class DistributionHist:
    def __init__(self, A1_data, bins, rng):
        self.range = rng
        self.bins = bins
        hist_data = _np.histogram(A1_data, bins=bins, range=rng, density=True)
        self.pdf = _scipy.stats.rv_histogram(hist_data, density=True).pdf


    def likelihood(self, x):
        return _np.prod(self.pdf(x))


def _prior(T):
    """From age of the globular cluster to present [Myr]."""
    if _fnc.numeric.within_equal(T, 0.0, 12_000.0):
        return 1.0/(12_000.0 - 0.0)
    return 0.0


def posterior(A1, T_eval, name_folder, prm_gc, progress=True):
    """Posterior distribution in function of 'T_eval' for a given data 'A1'."""

    #Load distribution properties
    bins = prm_gc['age_estimate']['bins']
    rng = prm_gc['age_estimate']['A1_rng']
    #bw = prm_gc['age_estimate']['bw']

    lkp = _np.zeros(len(T_eval))

    for i, T in enumerate(_tqdm.tqdm(T_eval, ncols=78)):
        #Load data to construct the distribution
        _, A1_sim, _, _ = load_surface_density_data(name_folder, T)

        #Definition distribution
        hist = DistributionHist(A1_sim, bins, rng)
        #hist = DistributionKDE(A1_sim, bw, rng)

        #Evaluation likelihood for the observational data
        lk = hist.likelihood(A1)

        #Add prior
        lkp[i] = lk*_prior(T)

    #Normalisation posterior
    post = lkp/_np.trapezoid(lkp, T_eval)

    return post

#-----------------------------------------------------------------------------
