"""Functions to generate a sample of mock observations.

Note
----
1)  Gaia selection function simulation:
    Data stored in /home/thalassa/.gaiaunlimited
    https://github.com/gaia-unlimited/gaiaunlimited
    https://gaiaunlimited.readthedocs.io/en/latest/index.html

2)  Perhaps the random selection has to be substituted by a cut in probability:
    take the stars with probability > x. This is deterministic.

3)  The process of star selection from a Gaia catalogue has to be incorporated
    to properly simulate the final selection (instead of magnitude and Dec cut)."""

import numpy as _np
import scipy as _scipy

import fnc as _fnc
_gaiaunlimited = _fnc.utils.lazy.Import("gaiaunlimited")
_astropy = _fnc.utils.lazy.Import("astropy")
_pygaia_errors = _fnc.utils.lazy.Import("pygaia.errors")
_pd = _fnc.utils.lazy.Import("pandas")

import invi as _invi
import invi.units as _un

__all__ = ["reddening", "selection_function", "create_file_extract", "selection", "uncertainties", "Sample"]

#-----------------------------------------------------------------------------

def reddening(s_icrs_dict, s_phot, cmd37_correction):
    """Reddening simulation."""
    bprp_red, g_red = _invi.photometry.reddening.simulation(s_icrs_dict['delta'],
                                                            s_icrs_dict['alpha'],
                                                            s_phot['bprp'],
                                                            s_phot['g'],
                                                            cmd37_correction)
    return {'g_red': g_red, 'bprp_red': bprp_red}

#-----------------------------------------------------------------------------

def _dr3(dec, ra, g):
    """Returns the probability to be observed by Gaia DR3 given the apparent
    magnitude g."""
    mapMulti = _gaiaunlimited.selectionfunctions.DR3SelectionFunctionTCG("multi")
    coordinates = _astropy.coordinates.SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
    probability = mapMulti.query(coordinates, g)
    return probability


def _dr3_rv(dec, ra, g, rp):
    """Returns the probability to observe radial velocity by Gaia DR3 given the
    apparent magnitudes g and rp."""
    rvssf = _gaiaunlimited.selectionfunctions.DR3RVSSelectionFunction()
    coordinates = _astropy.coordinates.SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
    probability = rvssf.query(coordinates, g=g, c=rp)
    return probability


def _random_selection(probability, random_state):
    """Randomly selects given the probability to be selected following a uniform distribution."""
    rnd_number = _scipy.stats.uniform.rvs(size=len(probability), loc=0.0, scale=1.0, random_state=random_state)
    selected = rnd_number < probability
    return selected


def selection_function(s_icrs_dict, s_phot, mock_red, random_state):
    #Definition dictionary
    sf = {}

    #Sky coordinates
    dec = s_icrs_dict['delta'] #[deg]
    ra = s_icrs_dict['alpha'] #[deg]

    #Probability observing star
    sf['prob_pass_sf'] = _dr3(dec, ra, mock_red['g_red'])

    #Probability observing star with radial velocity (Reddening not included)
    sf['prob_pass_sf_rv'] = _dr3_rv(dec, ra, s_phot['g'], s_phot['rp'])

    #Initialisation random generator
    rng = _np.random.default_rng(random_state)

    #Simulation observable stars by the selection function
    sf['pass_sel_func'] = _random_selection(sf['prob_pass_sf'], random_state=_invi.misc.seed(rng))
    sf['pass_sel_func_rv'] = _random_selection(sf['prob_pass_sf_rv'], random_state=_invi.misc.seed(rng))

    return sf

#-----------------------------------------------------------------------------
#Mock selection

def create_file_extract(sample_parallax, sample_icrs, s_unc):
    """Create file mocking gaia catalogue with extracted columns."""
    n = len(sample_parallax)
    z = _np.zeros(n)

    cosd = _np.cos(_invi.units.deg_to_rad(sample_icrs['delta']))

    data = {'parallax': sample_parallax,
            'dec': sample_icrs['delta'],
            'ra': sample_icrs['alpha'],
            'radial_velocity': sample_icrs['mu_r'],
            'pmdec': sample_icrs['mu_delta'],
            'pmra': sample_icrs['mu_alpha_str'],
            'parallax_error': s_unc['parallax'],
            'dec_parallax_corr': z,
            'ra_parallax_corr': z,
            'parallax_pmdec_corr': z,
            'parallax_pmra_corr': z,
            'dec_error': _invi.units.deg_to_mas(s_unc['ICRS']['delta']),
            'ra_dec_corr': z,
            'dec_pmdec_corr': z,
            'dec_pmra_corr': z,
            'ra_error': _invi.units.deg_to_mas(s_unc['ICRS']['alpha']*cosd),
            'ra_pmdec_corr': z,
            'ra_pmra_corr': z,
            'radial_velocity_error': s_unc['ICRS']['mu_r'],
            'pmdec_error': s_unc['ICRS']['mu_delta'],
            'pmra_pmdec_corr': z,
            'pmra_error': s_unc['ICRS']['mu_alpha_str'],
            'bp_rp': z,
            'phot_g_mean_mag': z}

    _pd.DataFrame(data).to_csv('/tmp/mock_stream_extract.csv', index=False, na_rep='NaN')


def selection(sample_parallax_dict, sample_icrs_dict, mock_sf, s_unc, s_phot, prm_gc, prm):
    #Create file mocking gaia catalogue with extracted columns.
    create_file_extract(sample_parallax_dict['parallax'], sample_icrs_dict, s_unc)

    file_data_extract = "/tmp/mock_stream_extract.csv"
    file_intersec = "/tmp/intersec.dat"
    file_orbit_bundle = "../../data/gaia_dr3/M68/selection/orbit_bundle.dat"

    #Count number lines 'file_data_extract'
    n_lines_file_data_extract = _invi.selection.count_file_lines(file_data_extract)

    #Compute intersections
    _invi.selection.intersections(n_lines_file_data_extract, file_data_extract, file_intersec, file_orbit_bundle)

    #Milky Way potential (galpy)
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Distance estimates
    ra = sample_icrs_dict['alpha']
    dec = sample_icrs_dict['delta']
    r_orb_est = _invi.stars.orbit_estimate.distance(ra, dec, prm_gc, prm, progress=True)

    BPRP_orb_est = s_phot['bprp']
    G_orb_est = _invi.photometry.magnitudes.m_to_M(s_phot['g'], r_orb_est)

    #Intersections
    intersec = _np.loadtxt("/tmp/intersec.dat", skiprows=1)

    #Selection cuts
    sample = {'parallax': sample_parallax_dict, 'ICRS': sample_icrs_dict}
    sel = _invi.selection.cuts(sample, BPRP_orb_est, G_orb_est, prm_gc, intersec)

    #GDR3 selection function
    sf = mock_sf['pass_sel_func']

    return sel & sf

#-----------------------------------------------------------------------------

def uncertainties(gdr, s_icrs_dict, synthetic_population, s_phot, mock_red):
    """Gaia catalogue error simulation: gdr = 'dr3', 'dr4', 'dr5'
    The uncertainties are computed for the g_red"""
    #---------------------------------------------
    def within_limits(x):
        """The functions pygaia.errors.photometric
        are only defined within the interval
        [4, 21] mag."""
        y = _np.copy(x)
        excl = (y < 4.0) | (y > 21.0) #[mag]
        y[excl] = _np.nan
        return y
    #---------------------------------------------
    #Data
    dec = s_icrs_dict['delta']

    #Teff = synthetic_population['Teff']
    Teff_red = _invi.photometry.magnitudes.BPRP_to_Teff(mock_red['bprp_red'])
    logg = synthetic_population['logg']

    g_red = mock_red['g_red']

    g = s_phot['g']
    bp = s_phot['bp']
    rp = s_phot['rp']

    #Definition dictionaries
    ICRS = {}
    phot = {}

    #Parallax
    parallax_uas = _pygaia_errors.astrometric.parallax_uncertainty(g_red, release=gdr) #[micro-arcseconds]
    parallax = _un.micro_to_milli( parallax_uas ) #[mas]

    #Sky coordinates
    ra_cosdelta_unc, delta_unc = _pygaia_errors.astrometric.position_uncertainty(g_red, release=gdr) #[micro-arcseconds]
    ICRS['alpha'] = _un.mas_to_deg( _un.micro_to_milli( ra_cosdelta_unc / _np.cos(_un.deg_to_rad(dec)) ) ) #[deg]
    ICRS['delta'] = _un.mas_to_deg( _un.micro_to_milli( delta_unc ) ) #[deg]

    #Radial velocity (I don't know if the observation of logg is affected by reddening)
    rv_unc = _pygaia_errors.spectroscopic.radial_velocity_uncertainty(g_red, Teff_red, logg, release=gdr) #[km/s]
    corr_factor = 2.0 #It is possible that it is only needed for EGDR3 but not for GDR3
    ICRS['mu_r'] = rv_unc * corr_factor

    #Proper motion
    mu_alpha_cosdelta_unc, mu_delta_unc = _pygaia_errors.astrometric.proper_motion_uncertainty(g_red, release=gdr) #[micro-arcseconds/yr]
    ICRS['mu_alpha_str'] = _un.micro_to_milli( mu_alpha_cosdelta_unc ) #[mas/yr]
    ICRS['mu_delta'] = _un.micro_to_milli( mu_delta_unc ) #[mas/yr]


    #Gaia magnitudes mock reddening
    g_red_unc = _pygaia_errors.photometric.magnitude_uncertainty(maglist=within_limits(g_red), band="g", release=gdr) #[mmag]
    phot['g_red'] = _un.milli_to_unit(g_red_unc) #[mag]

    #Gaia magnitudes
    g_unc = _pygaia_errors.photometric.magnitude_uncertainty(maglist=within_limits(g), band="g", release=gdr)
    phot['g'] = _un.milli_to_unit(g_unc)

    bp_unc = _pygaia_errors.photometric.magnitude_uncertainty(maglist=within_limits(bp), band="bp", release=gdr)
    phot['bp'] = _un.milli_to_unit(bp_unc)

    rp_unc = _pygaia_errors.photometric.magnitude_uncertainty(maglist=within_limits(rp), band="rp", release=gdr)
    phot['rp'] = _un.milli_to_unit(rp_unc)

    return {'parallax': parallax, 'ICRS': ICRS, 'photometry': phot}

#-----------------------------------------------------------------------------

def _norm_rvs(arr_loc, arr_scale, random_state):
    """Generate random samples when loc or scale are arrays including np.nan.

    Note
    ----
    1)  scipy.stats.norm.rvs is not defined for np.nan"""
    loc = _np.copy(arr_loc)
    scale = _np.copy(arr_scale)

    sel = _np.isnan(arr_loc) | _np.isnan(arr_scale)
    loc[sel] = 0.0
    scale[sel] = 0.0

    rvs = _scipy.stats.norm.rvs(loc=loc, scale=scale, random_state=random_state)
    rvs[sel] = _np.nan

    return rvs


class Sample:
    def __init__(self, s_icrs_dict, s_phot, s_unc, prm_gc, random_state):
        self.s_icrs_dict = s_icrs_dict
        self.s_phot = s_phot
        self.s_unc = s_unc
        self.prm_gc = prm_gc
        #self.random_state = random_state
        self.rng = _np.random.default_rng(random_state)


    def parallax(self):
        parallax = _invi.coordinates.r_to_parallax(self.s_icrs_dict['r']) #[mas]

        return _norm_rvs(arr_loc=parallax, arr_scale=self.s_unc['parallax'], random_state=_invi.misc.seed(self.rng))


    def ICRS(self):
        n = len(self.s_icrs_dict['r'])

        r = _np.array([_np.nan]*n)
        delta = _norm_rvs(self.s_icrs_dict['delta'], self.s_unc['ICRS']['delta'], _invi.misc.seed(self.rng))
        alpha = _norm_rvs(self.s_icrs_dict['alpha'], self.s_unc['ICRS']['alpha'], _invi.misc.seed(self.rng))
        mu_r = _norm_rvs(self.s_icrs_dict['mu_r'], self.s_unc['ICRS']['mu_r'], _invi.misc.seed(self.rng))
        mu_delta = _norm_rvs(self.s_icrs_dict['mu_delta'], self.s_unc['ICRS']['mu_delta'], _invi.misc.seed(self.rng))
        mu_alpha_str = _norm_rvs(self.s_icrs_dict['mu_alpha_str'], self.s_unc['ICRS']['mu_alpha_str'], _invi.misc.seed(self.rng))

        return _np.array([r, delta, alpha, mu_r, mu_delta, mu_alpha_str])


    def photometry_red(self):
        delta = _norm_rvs(self.s_icrs_dict['delta'], self.s_unc['ICRS']['delta'], _invi.misc.seed(self.rng))
        alpha = _norm_rvs(self.s_icrs_dict['alpha'], self.s_unc['ICRS']['alpha'], _invi.misc.seed(self.rng))

        g = _norm_rvs(self.s_phot['g'], self.s_unc['photometry']['g'], _invi.misc.seed(self.rng))
        bp = _norm_rvs(self.s_phot['bp'], self.s_unc['photometry']['bp'], _invi.misc.seed(self.rng))
        rp = _norm_rvs(self.s_phot['rp'], self.s_unc['photometry']['rp'], _invi.misc.seed(self.rng))

        bprp = bp - rp

        cmd37_correction = self.prm_gc['mock']['cmd37_correction']
        bprp_red, g_red = _invi.photometry.reddening.simulation(delta, alpha, bprp, g, cmd37_correction)

        return {'bprp_red': bprp_red, 'g_red': g_red}

#-----------------------------------------------------------------------------
