"""Functions to estimate the age of an stellar stream."""

import numpy as _np
import scipy as _scipy

import fnc as _fnc
import invi as _invi

_tqdm = _fnc.utils.lazy.Import("tqdm")
_os = _fnc.utils.lazy.Import("os")
_fnmatch = _fnc.utils.lazy.Import("fnmatch")

__all__ = ["ages",
           "number_samples", "load_samples", "surface_density", "number_stars",
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

def _count_files(directory, pattern):
    """Count number of files in a directory matching some pattern."""
    return len(_fnmatch.filter(_os.listdir(directory), pattern))


def number_samples(name_folder):
    return _count_files(name_folder, 'sample_[0-9]*')


def load_samples(name_folder):
    """Load samples."""
    n_samples = number_samples(name_folder)

    sim_AAF = [[]]*n_samples
    A1_orb_est = [[]]*n_samples
    sel = [[]]*n_samples
    magnitude_cut = [[]]*n_samples

    for i in range(n_samples):
        name_file = f"{name_folder}/sample_{i}.npz"

        sim_AAF[i] = _np.load(name_file)['sim_AAF']
        A1_orb_est[i] = _np.load(name_file)['A1_orb_est']
        sel[i] = _np.load(name_file)['sel']
        magnitude_cut[i] = _np.load(name_file)['magnitude_cut']

    arr_sim_AAF = _np.concatenate(sim_AAF, axis=1)
    sim_stripping_time = -_invi.inverse.integration_time(arr_sim_AAF)

    data = {'sim_AAF': arr_sim_AAF,
            'A1_orb_est': _np.concatenate(A1_orb_est),
            'sel': _np.concatenate(sel),
            'magnitude_cut': _np.concatenate(magnitude_cut),
            'sim_stripping_time': sim_stripping_time,
            'n_samples': n_samples}

    return data


def surface_density(data, T):
    """Compute surface density of age 'T' in Myr."""

    #Select stars
    selected = data['sel'] & (data['sim_stripping_time'][data['magnitude_cut']] >= -T)

    A1_sim = data['A1_orb_est'][selected]

    return A1_sim


def number_stars(name_folder, T):
    """Number of stars in the samples."""

    n_samples = number_samples(name_folder)

    n_stars = []

    for i in range(n_samples):
        name_file = f"{name_folder}/sample_{i}.npz"

        sim_AAF = _np.load(name_file)['sim_AAF']
        A1_orb_est = _np.load(name_file)['A1_orb_est']
        sel = _np.load(name_file)['sel']
        magnitude_cut = _np.load(name_file)['magnitude_cut']

        #Stripping time of each star
        sim_stripping_time = -_invi.inverse.integration_time(sim_AAF) #[Myr]

        #sel_sim and stripping time cuts
        selected = sel & (sim_stripping_time[magnitude_cut] >= -T)

        #Number of stars of each sample
        n_stars.append(len(A1_orb_est[selected]))

    return n_stars

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

    #Load samples
    data = load_samples(name_folder)

    lkp = _np.zeros(len(T_eval))

    for i, T in enumerate(_tqdm.tqdm(T_eval, ncols=78, disable=not progress)):
        #Load data to construct the distribution
        A1_sim = surface_density(data, T)

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
