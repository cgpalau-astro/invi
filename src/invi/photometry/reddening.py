"""Reddening correction for Gaia G magnitude and BPRP colour index using the
SFD model from the package dustmaps.

Note
----
1)  The reddening correction can be applied to apparent (m) and absolute magnitude (M).

2)  Given a magnitude m:
    Reddening simulated: m_red

    Given observed magnitude: m_red
    Reddening corrected: m"""

import fnc as _fnc
_units = _fnc.utils.lazy.Import("astropy.units")
_crd = _fnc.utils.lazy.Import("astropy.coordinates")
_sfd = _fnc.utils.lazy.Import("dustmaps.sfd")

import invi as _invi

__all__ = ["correction", "simulation"]

#-----------------------------------------------------------------------------

def _reddening_corr_factor(dec, ra, cmd37_correction=0.0):
    """Reddening correction factor: E(G_BP - G_RP) = E(BPRP) #[mag]

    Note
    ----
    1)  The factor 'cmd37_correction' is included to improve the fit with the
        CMD3.7 model of the M68 globular cluster."""
    coords = _crd.SkyCoord(ra=ra*_units.deg, dec=dec*_units.deg, frame='icrs')
    sfd = _sfd.SFDQuery()
    EBV = sfd(coords)
    E_BPRP = _invi.photometry.magnitudes.BV_to_BPRP_approx(0.86*EBV + cmd37_correction)
    return E_BPRP

#-----------------------------------------------------------------------------

def correction(dec, ra, BPRP_red, G_red, cmd37_correction):
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
    cmd37_correction : float
        Factor to improve the fit with the cmd37 synthetic population [mag]

    Returns
    -------
    BPRP : float
        Gaia BPRP colour index reddening corrected [mag]
    G : float
        Gaia G magnitude reddening corrected [mag]"""
    red = _reddening_corr_factor(dec, ra, cmd37_correction)
    BPRP = BPRP_red - red
    G = G_red - 1.98*red
    return BPRP, G

#-----------------------------------------------------------------------------

def simulation(dec, ra, BPRP, G, cmd37_correction):
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
    cmd37_correction : float
        Factor to improve the fit with the cmd37 synthetic population [mag]

    Returns
    -------
    BPRP_red : float
        Gaia BPRP colour index with reddening simulated [mag]
    G_red : float
        Gaia G magnitude with reddening simulated [mag]"""
    red = _reddening_corr_factor(dec, ra, cmd37_correction)
    BPRP_red = BPRP + red
    G_red = G + 1.98*red
    return BPRP_red, G_red

#-----------------------------------------------------------------------------
