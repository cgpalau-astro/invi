"""Photometry of the stars."""

import invi as _invi

__all__ = ["photometry"]

#-----------------------------------------------------------------------------

def photometry(s_icrs_dict, synthetic_population):
    """Photometry of stars from the synthetic population."""

    #Heliocentric distance of the source.
    distance = s_icrs_dict['r'] #[kpc]

    return {#Apparent magnitudes [mag]
            'g': _invi.photometry.magnitudes.M_to_m(synthetic_population['G'], distance),
            'bp': _invi.photometry.magnitudes.M_to_m(synthetic_population['BP'], distance),
            'rp': _invi.photometry.magnitudes.M_to_m(synthetic_population['RP'], distance),
            'bprp': synthetic_population['BPRP'],
            #Absolute magnitudes [mag]
            'G': synthetic_population['G'],
            'BP': synthetic_population['BP'],
            'RP': synthetic_population['RP'],
            'BPRP': synthetic_population['BPRP']}

#-----------------------------------------------------------------------------
