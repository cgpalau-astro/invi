"""General functions for plots."""

import numpy as _np

import fnc as _fnc
_pl = _fnc.utils.lazy.Import("pathlib")

__all__ = ["lines_b15"]

#-----------------------------------------------------------------------------

def lines_b15(ax, zorder=0, linewidth=1.0):
    """Plot limits of the disc at b=±15 deg and Galactic centre in ICRS."""
    base_path = _pl.Path(__file__).parent
    file_path = (base_path / "ICRS_b_15deg.csv.gz").resolve()

    kwargs = {'color': "k", 'alpha': 0.5, 'zorder': zorder}

    #Disc b=±15 deg lines
    alpha_0, delta_0, alpha_1, delta_1 = _np.loadtxt(file_path, delimiter=',', skiprows=4, unpack=True)
    ax.plot(alpha_0, delta_0, linestyle='--', linewidth=linewidth, **kwargs)
    ax.plot(alpha_1, delta_1, linestyle='--', linewidth=linewidth, **kwargs)

    #Galactic centre Sgr A*
    ax.scatter(266.4167, -29.0078, s=25.0, marker='x', linewidth=linewidth-0.25, **kwargs)

#-----------------------------------------------------------------------------
