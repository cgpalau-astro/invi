"""Plots for Jupyter notebooks in code/11-radial_velocities folder."""

import numpy as _np

import invi as _invi
import fnc as _fnc

_pl = _fnc.utils.lazy.Import("pathlib")
_matplotlib_colors = _fnc.utils.lazy.Import("matplotlib.colors")

__all__ = ["sky_hist", "sky_vr"]

#-----------------------------------------------------------------------------

def _gc(ax, prm_gc):
    ax.scatter(prm_gc['ICRS']['alpha'], prm_gc['ICRS']['delta'], s=15.0, c='w')
    ax.scatter(prm_gc['ICRS']['alpha'], prm_gc['ICRS']['delta'], s=10.0, c='k')


def _arr_to_str(arr):
    return [f"{_np.int64(item)}" for item in arr]


def _gc_equal_area(ax, prm_gc):
    ra, dec = _invi.coordinates.radec_to_equal_area(prm_gc['ICRS']['alpha'], prm_gc['ICRS']['delta'])
    ax.scatter(ra, dec, s=15.0, c='w')
    ax.scatter(ra, dec, s=10.0, c='k')


def _lines_b15_equal_area(ax, zorder=0):
    """Plot limits of the disc at b=±15 deg in ICRS."""
    base_path = _pl.Path(__file__).parent.parent
    file_path = (base_path / "plot/ICRS_b_15deg.csv.gz").resolve()

    alpha_0, delta_0, alpha_1, delta_1 = _np.loadtxt(file_path, delimiter=",", skiprows=4, unpack=True)

    alpha_0, delta_0 = _invi.coordinates.radec_to_equal_area(alpha_0, delta_0)
    alpha_1, delta_1 = _invi.coordinates.radec_to_equal_area(alpha_1, delta_1)

    kwargs = {'color': "k", 'linestyle': "--", 'alpha': 0.5, 'zorder': zorder}
    ax.plot(alpha_0, delta_0, **kwargs)
    ax.plot(alpha_1, delta_1, **kwargs)


#-----------------------------------------------------------------------------
#Sky maps

def sky_hist(alpha, delta, prm_gc, fc_bins=100):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(3.75, 1.75))

    bins = _np.array([36, 18], dtype=_np.int64)*fc_bins

    H = _np.histogram2d(_np.asarray(alpha), _np.asarray(delta),
                        bins=bins,
                        range=[[0.0, 360.0], [-90.0, 90.0]])[0]

    cbar = ax.imshow(H.T,
                     origin='lower',
                     extent=[0.0, 360.0, -90.0, 90.0],
                     cmap='turbo',
                     zorder=0,
                     norm=_matplotlib_colors.LogNorm(vmin=0.9999, vmax=_np.max(H)))

    cbar.cmap.set_under((1,1,1))

    _gc(ax, prm_gc)

    ax.set_aspect(1.0)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")
    _invi.plot.lines_b15(ax)

    return ax


def sky_hist_equal_area(ra, dec, prm_gc, fc_bins=500):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(3.75, 1.75))

    ea_1, ea_2 = _invi.coordinates.radec_to_equal_area(ra, dec)

    bins = _np.array([4, 2], dtype=_np.int64)*fc_bins

    H = _np.histogram2d(ea_1, ea_2,
                        bins=bins,
                        range=[[0.0, 4.0], [-1.0, 1.0]])[0]

    cbar = ax.imshow(H.T,
                     origin='lower',
                     extent=[0.0, 4.0, -1.0, 1.0],
                     cmap='turbo',
                     zorder=0,
                     norm=_matplotlib_colors.LogNorm(vmin=0.9999, vmax=_np.max(H))
                     )

    cbar.cmap.set_under((1,1,1))

    _lines_b15_equal_area(ax, zorder=1)

    _gc_equal_area(ax, prm_gc)

    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect(1.0)

    xintv = 30.0 #[deg]
    yintv = 30.0 #[deg]

    xticks = _np.arange(0.0, 360.0 + xintv, xintv)
    yticks = _np.arange(-90.0, 90.0 + yintv, yintv)

    xticks_ea, yticks_ea = _invi.coordinates.radec_to_equal_area(xticks, yticks)

    ax.set_xticks(xticks_ea)
    ax.set_xticklabels(_arr_to_str(xticks))

    ax.set_yticks(yticks_ea)
    ax.set_yticklabels(_arr_to_str(yticks))

    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")

    return ax

#-----------------------------------------------------------------------------

def sky_vr(alpha, delta, mu_r,
           prm_gc, limit=250.0 #[km/s]
           ):
    fig, ax = _fnc.plot.figure(fc=(3.75, 1.75))

    cbar = ax.scatter(_np.asarray(alpha),
                      _np.asarray(delta),
                      c=_np.asarray(mu_r),
                      s=1.0E-2,
                      cmap='seismic', #'bwr', 'coolwarm', 'RdBu'
                      vmin=-limit,
                      vmax=limit)

    fig.colorbar(cbar, label='Vr [km/s]', extend='both')

    _gc(ax, prm_gc)

    ax.set_aspect(1.0)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")
    _invi.plot.lines_b15(ax, zorder=2)

    return ax

#-----------------------------------------------------------------------------
