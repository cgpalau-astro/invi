"""Distance, radial velocity, and angle along the principal axis of the stream
estimation from the orbit of the globular cluster."""

import numpy as _np
import scipy as _scipy
import tqdm as _tqdm

import invi as _invi

__all_ = ["distance", "radial_velocity", "A1"]

#-----------------------------------------------------------------------------

def _spline_def(x):
    """Spline definition"""
    tau = _np.linspace(0.0, 1.0, len(x))
    return _scipy.interpolate.make_interp_spline(tau, x)


def _spline_def_eval(x, tau):
    """Spline definition and evaluation"""
    spl = _spline_def(x)
    return spl(tau)


class _DefineOrbitSpline:
    def __init__(self, orb_icrs):
        self.ra  = _spline_def(orb_icrs[2]) #alpha [deg]
        self.dec = _spline_def(orb_icrs[1]) #delta [deg]

#-----------------------------------------------------------------------------

def _orbit_gc_icrs(prm_gc, prm):
    #Milky Way potential (galpy)
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Initial condition
    gc_icrs = _invi.dicts.dict_to_array(prm_gc['ICRS'])
    gc_fsr = _invi.coordinates.ICRS_to_FSR(gc_icrs, prm['sun'], mw.v_lsr)

    #Forward orbit
    T = prm_gc['orbit_estimates']['T']
    N = prm_gc['orbit_estimates']['N']
    _t, orb_fsr = _invi.galpy.orbit.integrate(gc_fsr, mw.potential, T, N)
    orb_icrs = _invi.coordinates.FSR_to_ICRS(orb_fsr, prm['sun'], mw.v_lsr)

    return orb_icrs


def _orbit_gc_AAF(prm_gc, prm):
    #Milky Way potential (galpy)
    mw = _invi.galpy.potential.DefineGalaxy(prm['sun']['R'], prm['mw'])

    #Angle, action, frequency accuracy
    b, maxn, tintJ, ntintJ = prm['isochrone_approx']['invi'].values()

    #Initial condition
    gc_icrs = _invi.dicts.dict_to_array(prm_gc['ICRS'])
    gc_fsr = _invi.coordinates.ICRS_to_FSR(gc_icrs, prm['sun'], mw.v_lsr)
    gc_aaf = _invi.coordinates.FSR_to_aaf(gc_fsr, mw.potential, b, maxn, tintJ, ntintJ)

    #Forward orbit
    T = prm_gc['orbit_estimates']['T']
    N = prm_gc['orbit_estimates']['N']
    _t, orb_fsr = _invi.galpy.orbit.integrate(gc_fsr, mw.potential, T, N)

    #Angle actions isochrone approximation
    orb_aaf = _invi.coordinates.FSR_to_aaf(orb_fsr, mw.potential, b, maxn, tintJ, ntintJ, n_cpu=None, progress=False)
    orb_AAF = _invi.coordinates.aaf_to_AAF(orb_aaf, gc_aaf, prm['M68']['stream']['varphi'])

    return orb_AAF

#-----------------------------------------------------------------------------

def _min_distance(tau, ra, dec, spline):
    """Angular distance."""
    if 0.0 < tau < 1.0:
        return _invi.misc.angular_distance(ra, dec, spline.ra(tau), spline.dec(tau))
    return _np.inf


def _minimum_tau(ra, dec, orb_icrs, prm_gc, progress):
    """Estimate parameter tau corresponding to the minimum distance from the orbit to each star."""

    #Globular cluster section orbit sky-coordinates described with a spline
    orb_spline = _DefineOrbitSpline(orb_icrs)

    n = len(ra)
    tau = _np.zeros(n)
    for i in _tqdm.tqdm(range(n), ncols=78, disable=not progress):
        #Leading arm within limits [deg]
        if (ra[i] < 300.0) & (dec[i] > -20.0):
            result = _scipy.optimize.minimize(_min_distance,
                                              x0=0.4,
                                              args=(ra[i], dec[i], orb_spline),
                                              method="Nelder-Mead",
                                              options={'disp': False, 'maxfev':1_000},
                                              tol=prm_gc['orbit_estimates']['tol'])
            tau[i] = result.x[0]
        else:
            tau[i] = _np.nan

    return tau

#-----------------------------------------------------------------------------

def distance(ra, dec, prm_gc, prm, progress=False):
    """Heliocentric distance (spherical r) in kpc estimated from the obit of the cluster."""

    #Orbit globular cluster
    orb_icrs = _orbit_gc_icrs(prm_gc, prm)

    #Minimum angular distance corresponding to each star
    tau = _minimum_tau(ra, dec, orb_icrs, prm_gc, progress)

    #Distance estimates [kpc]
    r = _spline_def_eval(orb_icrs[0], tau)

    #Apply correction factors
    r += prm_gc['orbit_estimates']['r_corr']

    return r


def radial_velocity(ra, dec, prm_gc, prm, progress=False):
    """Radial velocity (spherical mu_r) in km/s estimated from the obit of the cluster."""

    #Orbit globular cluster
    orb_icrs = _orbit_gc_icrs(prm_gc, prm)

    #Minimum angular distance corresponding to each star
    tau = _minimum_tau(ra, dec, orb_icrs, prm_gc, progress)

    #Radial velocity estimates [km/s]
    rv = _spline_def_eval(orb_icrs[3], tau)

    #Apply correction factors
    rv += prm_gc['orbit_estimates']['rv_corr']

    return rv


def A1(ra, dec, prm_gc, prm, progress=False):
    """Angle in principal axes reference frame in rad estimated from the obit of the cluster."""

    #Orbit globular cluster
    orb_icrs = _orbit_gc_icrs(prm_gc, prm)
    orb_AAF = _orbit_gc_AAF(prm_gc, prm)

    #Minimum angular distance corresponding to each star
    tau = _minimum_tau(ra, dec, orb_icrs, prm_gc, progress)

    #Angle estimates [rad]
    a1 = _spline_def_eval(orb_AAF[0], tau)

    #Apply correction factors [rad]
    a1 += prm_gc['orbit_estimates']['A1_corr']

    return a1

#-----------------------------------------------------------------------------

