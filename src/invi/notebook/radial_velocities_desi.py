"""Plots for Jupyter notebooks in code/11-radial_velocities folder."""

import numpy as _np

import invi as _invi
import fnc as _fnc

_pl = _fnc.utils.lazy.Import("pathlib")
_matplotlib_colors = _fnc.utils.lazy.Import("matplotlib.colors")

__all__ = ["sky_hist", "sky_hist_equal_area",
           "sky_hist_phi", "sky_hist_equal_area_phi",
           "sky_vr",
           "phi_1_phi_2", "phi_mu_r", "cmd",
           "FeH_AlphaFe_hist", "FeH_AlphaFe", "Vsini_hist"]

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

def sky_hist(alpha, delta, prm_gc, fc_bins=100, lines_b15=True):
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

    if lines_b15:
        _invi.plot.lines_b15(ax)

    ax.set_aspect(1.0)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")

    return ax


def sky_hist_equal_area(ra, dec, prm_gc, fc_bins=500, lines_b15=True):
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

    if lines_b15:
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
#Sky maps phi

def sky_hist_phi(alpha, delta, prm_gc, fc_bins=100):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(3.75, 1.75))

    bins = _np.array([36, 18], dtype=_np.int64)*fc_bins

    H = _np.histogram2d(_np.asarray(alpha), _np.asarray(delta),
                        bins=bins,
                        range=[[-60.0, 360.0], [-90.0, 90.0]])[0]

    cbar = ax.imshow(H.T,
                     origin='lower',
                     extent=[-60.0, 360.0, -90.0, 90.0],
                     cmap='turbo',
                     zorder=0,
                     norm=_matplotlib_colors.LogNorm(vmin=0.9999, vmax=_np.max(H)))

    cbar.cmap.set_under((1,1,1))

    _gc(ax, prm_gc)

    ax.set_aspect(1.0)
    ax.set_xlim(-60.0, 300.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("$\\phi_1$ [deg]")
    ax.set_ylabel("$\\phi_2$ [deg]")

    return ax


def sky_hist_equal_area_phi(ra, dec, prm_gc, fc_bins=500):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(3.75, 1.75))

    ea_1, ea_2 = _invi.coordinates.radec_to_equal_area(ra, dec)

    bins = _np.array([4, 2], dtype=_np.int64)*fc_bins

    eps = 2.0/3.0

    H = _np.histogram2d(ea_1, ea_2,
                        bins=bins,
                        range=[[-eps, 4.0-eps], [-1.0, 1.0]])[0]

    cbar = ax.imshow(H.T,
                     origin='lower',
                     extent=[-eps, 4.0-eps, -1.0, 1.0],
                     cmap='turbo',
                     zorder=0,
                     norm=_matplotlib_colors.LogNorm(vmin=0.9999, vmax=_np.max(H))
                     )

    cbar.cmap.set_under((1,1,1))

    _gc_equal_area(ax, prm_gc)

    ax.set_xlim(-eps, 4.0-eps)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect(1.0)

    xintv = 30.0 #[deg]
    yintv = 30.0 #[deg]

    xticks = _np.arange(-60.0, 300.0 + xintv, xintv)
    yticks = _np.arange(-90.0, 90.0 + yintv, yintv)

    xticks_ea, yticks_ea = _invi.coordinates.radec_to_equal_area(xticks, yticks)

    ax.set_xticks(xticks_ea)
    ax.set_xticklabels(_arr_to_str(xticks))

    ax.set_yticks(yticks_ea)
    ax.set_yticklabels(_arr_to_str(yticks))

    ax.set_xlabel("$\\phi_1$ [deg]")
    ax.set_ylabel("$\\phi_2$ [deg]")

    return ax

#-----------------------------------------------------------------------------

def sky_vr(alpha, delta, mu_r,
           prm_gc, limit=250.0, #[km/s]
           lines_b15=True):

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

    if lines_b15:
        _invi.plot.lines_b15(ax, zorder=2)

    ax.set_aspect(1.0)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")

    return ax

#-----------------------------------------------------------------------------

def phi_1_phi_2(ax, phi_1, phi_2, sel):
    ax.scatter(phi_1[sel], phi_2[sel], s=5.0, c='k', label='Sel')

    ax.legend()
    ax.set_aspect(1)
    ax.set_xlim(-40.0, 120.0)
    ax.set_ylim(-20.0, 0.0)
    ax.set_xlabel("$\\phi_1$ [deg]")
    ax.set_ylabel("$\\phi_2$ [deg]")


def phi_mu_r(ax, phi_1, mu_r, mu_r_unc, sel):

    ax.scatter(phi_1[sel], mu_r[sel], s=5.0, c='k', label='Sel')
    ax.errorbar(phi_1[sel], mu_r[sel], yerr=mu_r_unc[sel], c="k", linestyle='')

    ax.legend()

    ax.set_xlim(-40.0, 120.0)
    ax.set_ylim(-200.0, 300.0)
    ax.set_xlabel("$\\phi_1$ [deg]")
    ax.set_ylabel("$Vr$ [km/s]")


def cmd(ax, sp_GR, sp_R, GR, R, sel, prm, legend=False):
    #Synthetic population
    ax.scatter(sp_GR, sp_R, s=1.0, c='g', label=f"Synth. Pop. {prm['M68']['ICRS']['r']} kpc")

    #Pre-selection cross-match DESI
    #ax.scatter(GR_red[sel], R[sel], s=2.0, c="r", label='pre-sel-DESI red')
    ax.scatter(GR[sel], R[sel], s=2.0, c="k", label='pre_sel-DESI red corrected')

    if legend:
        ax.legend()

    ax.set_xlabel("$G-R$ [mag]")
    ax.set_ylabel("$R$ [mag]")
    ax.set_xlim(-1.0, 2.5)
    ax.set_ylim(12.0, -4.0)


def FeH_AlphaFe_hist(ax, FeH, AlphaFe, sel):

    kwargs = {'range': [-4.0, 2.0], 'histtype': 'step', 'bins': 100, 'density': True}

    ax.hist(FeH, color='r', alpha=0.2, **kwargs, label='FeH All')
    ax.hist(AlphaFe, color='b', alpha=0.2, **kwargs, label='AlphaFe All')

    ax.hist(FeH[sel], color='r', **kwargs, label='FeH Sel')
    ax.hist(AlphaFe[sel], color='b', **kwargs, label='AlphaFe Sel')

    ax.legend()

    ax.set_xlim(-4.0, 2.0)
    ax.set_xlabel('FeH [mag]')
    ax.set_ylabel('AlphaFe [mag]')


def FeH_AlphaFe(ax, FeH, AlphaFe, sel, sel_met, coef, eps):

    kwargs = {'s': 1.0}

    ax.scatter(FeH, AlphaFe, color='g', **kwargs, label='All')
    ax.scatter(FeH[sel_met], AlphaFe[sel_met], color='b', **kwargs, label='Met')
    ax.scatter(FeH[sel], AlphaFe[sel], color='r', **kwargs, label='Sel')

    ax.legend()

    x = _np.linspace(-4.0, 1.0, 1000)
    y = _np.polynomial.Polynomial(coef)(x)
    ax.plot(x, y+eps, c='b')
    ax.plot(x, y, c='b', linestyle='--')
    ax.plot(x, y-eps, c='b')

    ax.set_xlabel('FeH [mag]')
    ax.set_ylabel('AlphaFe [mag]')
    ax.set_xlim(-4.0, 1.0)
    ax.set_ylim(-0.75, 1.5)


def Vsini_hist(ax, Vsini, sel):
    kwargs = {'range': [0.005, 3E-2], 'histtype': 'step', 'bins': 100, 'density': True}

    ax.hist(Vsini, color='g', **kwargs, label='All')
    ax.hist(Vsini[sel], color='r', **kwargs, label='Sel')

    ax.legend()
    ax.set_xlabel("Vsini []")
    ax.set_yscale('log')

#-----------------------------------------------------------------------------
