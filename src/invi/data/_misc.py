"""Load miscellaneous catalogues."""

import termcolor as _tc
import numpy as _np

import fnc as _fnc
_pd = _fnc.utils.lazy.Import("pandas")

import invi as _invi

__all__ = ["csv", "sos", "ibata"]

#-----------------------------------------------------------------------------

def _print_info(df, name):
    #Number stars
    n_stars = df.shape[0]

    #Memory used by the data frame
    used_memory_B = df.memory_usage(deep=True).sum()
    used_memory = _fnc.utils.human_readable.memory(used_memory_B)

    _tc.cprint(f"{name}:", 'light_blue')
    print(f"Number stars = {n_stars:_}")
    print(f" Used memory = {used_memory}")

#-----------------------------------------------------------------------------

class csv:
    @staticmethod
    def load(file_name, verbose=True):
        """Load data from a csv file and return it as a dictionary of np.arrays."""

        df = _pd.read_csv(file_name).fillna(_np.nan)
        di = df.to_dict('list')

        for key in di.keys():
            di[key] = _np.asarray(di[key])

        #----------------------------------------
        if verbose:
            _print_info(df, 'csv')
        #----------------------------------------

        return di

#-----------------------------------------------------------------------------

class sos:
    @staticmethod
    def load(file_name, verbose=True):
        """Load stars from Survey of Surveys DR1 catalogue."""

        df = _pd.read_csv(file_name).fillna(_np.nan)

        #Definition dictionary
        stars = {}
        stars['source_id'] = df.sosdr1_gaiaSourceId.values
        stars['survey'] = df.sosdr1_surveysId.values

        stars['ICRS'] = {'delta': df.sosdr1_decl.values,
                         'alpha': df.sosdr1_ra.values,
                         'mu_r': df.sosdr1_RVcor_merged.values}

        stars['uncertainties'] = {'ICRS': {'mu_r': df.sosdr1_errRVcor_merged.values}}

        #----------------------------------------
        if verbose:
            _print_info(df, 'Survey of surveys')
        #----------------------------------------

        return stars

#-----------------------------------------------------------------------------

class ibata:
    @staticmethod
    def load(file_name, cmd37_correction, verbose=True):
        """Load stars from Ibata Streamfinder catalogue."""

        df = _pd.read_csv(file_name).fillna(_np.nan)

        #M68 stream
        st = df.Stream.values == 22

        #Definition dictionary
        stars = {}
        stars['source_id'] = df.Gaia.values[st]
        stars['parallax'] = df.plx.values[st]

        stars['ICRS'] = {'r': df.dSF.values[st], #Distance estimated by Streamfinder
                         'delta': df.DEJ2000.values[st],
                         'alpha': df.RAJ2000.values[st],
                         'mu_r': df.HRV.values[st],
                         'mu_delta': df.pmDE.values[st],
                         'mu_alpha_str': df.pmRA.values[st]}

        stars['uncertainties'] = {'ICRS': {'mu_r': df.e_HRV.values[st]}}

        #Photometry
        stars['photometry_red'] = {'g_red': df.Gmag0.values[st],
                                   'bprp_red': df['(BP-RP)0'].values[st]}

        bprp, g = _invi.photometry.reddening.correction(stars['ICRS']['delta'], stars['ICRS']['delta'],
                                                        stars['photometry_red']['bprp_red'], stars['photometry_red']['g_red'],
                                                        cmd37_correction)

        stars['photometry'] = {'g': g,
                               'bprp': bprp,
                               'G': _invi.photometry.magnitudes.m_to_M(g, stars['ICRS']['r']),
                               'BPRP': bprp}

        #----------------------------------------
        if verbose:
            _print_info(df, 'Ibata streamfinder')
        #----------------------------------------

        return stars

#-----------------------------------------------------------------------------
