"""Reddening correction for Gaia G magnitude and BPRP colour index using the
SFD model from the package dustmaps.

Note
----
1)  The reddening correction can be applied to apparent (m) and absolute magnitude (M).

2)  Given a magnitude m:
    Reddening simulated: m_red

    Given observed magnitude: m_red
    Reddening corrected: m"""

import numpy as _np

import fnc as _fnc
_units = _fnc.utils.lazy.Import("astropy.units")
_crd = _fnc.utils.lazy.Import("astropy.coordinates")
_sfd = _fnc.utils.lazy.Import("dustmaps.sfd")

import invi as _invi

__all__ = ["correction", "simulation",
           "correction_DESI", "simulation_DESI"]

#-----------------------------------------------------------------------------

def _reddening_corr_factor(dec, ra, correction, factor=0.86):
    """Reddening correction factor: E(B-V) #[mag]

    Note
    ----
    1)  The 'correction' is included to improve the fit with the CMD3.7
        synthetic population.
    2)  The 'factor' is included following the recalibration of Schlafly &
        Finkbeiner (2011)."""

    #Take only the stars within 'ra' and 'dec' limits)
    if isinstance(ra, float) & isinstance(ra, float):
        c_ra = ra
        c_dec = dec
        if not (_fnc.numeric.within_equal(ra, 0.0, 360.0) & _fnc.numeric.within_equal(dec, -90.0, 90.0)):
            c_ra = _np.nan
            c_dec = _np.nan
    else:
        c_ra = ra.copy()
        c_dec = dec.copy()
        for i in range(len(c_ra)):
            if not (_fnc.numeric.within_equal(c_ra[i], 0.0, 360.0) & _fnc.numeric.within_equal(c_dec[i], -90.0, 90.0)):
                c_ra[i] = _np.nan
                c_dec[i] = _np.nan

    coords = _crd.SkyCoord(ra=c_ra*_units.deg, dec=c_dec*_units.deg, frame='icrs')
    sfd = _sfd.SFDQuery()
    EBV = sfd(coords)
    E_BV = factor*EBV + correction
    return E_BV

#-----------------------------------------------------------------------------

def correction(dec, ra, BPRP_red, G_red, correction):
    """Eliminate the reddening effect on the BPRP and G magnitudes.

    Parameters
    ----------
    dec : float
        Declination [deg]
    ra : float
        Right ascension [deg]
    BPRP_red : float
        Gaia BPRP colour index with reddening [mag]
    G_red : float
        Gaia G magnitude with reddening  [mag]
    correction : float
        Factor to improve the fit with the cmd37 synthetic population [mag]

    Returns
    -------
    BPRP : float
        Gaia BPRP colour index reddening corrected [mag]
    G : float
        Gaia G magnitude reddening corrected [mag]"""
    E_BV = _reddening_corr_factor(dec, ra, correction)
    E_BPRP = _invi.photometry.magnitudes.BV_to_BPRP_approx(E_BV)
    BPRP = BPRP_red - E_BPRP
    G = G_red - 1.98*E_BPRP
    return BPRP, G

#-----------------------------------------------------------------------------

def simulation(dec, ra, BPRP, G, correction):
    """Simulate the reddening effect on the BPRP and g magnitudes.

    Parameters
    ----------
    dec : float
        Declination [deg]
    ra : float
        Right Ascension [deg]
    BPRP : float
        Gaia BPRP colour index [mag]
    G : float
        Gaia G magnitude [mag]
    correction : float
        Factor to improve the fit with the cmd37 synthetic population [mag]

    Returns
    -------
    BPRP_red : float
        Gaia BPRP colour index with reddening simulated [mag]
    G_red : float
        Gaia G magnitude with reddening simulated [mag]"""
    E_BV = _reddening_corr_factor(dec, ra, correction)
    E_BPRP = _invi.photometry.magnitudes.BV_to_BPRP_approx(E_BV)
    BPRP_red = BPRP + E_BPRP
    G_red = G + 1.98*E_BPRP
    return BPRP_red, G_red

#-----------------------------------------------------------------------------

def correction_DESI(dec, ra, GR_red, correction):
    """Eliminate the reddening effect on the DESI GR (g-r) colour index [mag].

    Note
    ----
    1)  arXiv: 2409.05140v3"""
    E_BV = _reddening_corr_factor(dec, ra, correction, factor=1.0)
    GR = GR_red - 1.049*E_BV
    return GR


def simulation_DESI(dec, ra, GR, correction):
    """Simulate the reddening effect on the DESI GR (g-r) colour index [mag].

    Note
    ----
    1)  arXiv: 2409.05140v3"""
    E_BV = _reddening_corr_factor(dec, ra, correction, factor=1.0)
    GR_red = GR + 1.049*E_BV
    return GR_red

#-----------------------------------------------------------------------------
