"""Plots for Jupyter notebooks in code/7-mock folder."""

import fnc as _fnc
import invi as _invi

__all__ = ["HR", "set_axes_sky_coord"]

#-----------------------------------------------------------------------------

def _set_axis(ax, gaia_limit, x_limit, y_limit):
    ax.plot(x_limit, [gaia_limit, gaia_limit], linestyle="--", color='0.7')

    #Lower x axis
    ax.set_xlabel("$(g_{\\rm BP}-g_{\\rm RP})$ [mag]")
    ax.set_xlim(x_limit)

    #Upper x axis
    ax2 = ax.twiny()
    ax2.set_xlabel("$T_{\\rm eff}$ [K]")
    ax2.set_xlim(x_limit)

    xticks = []
    xticklabels = []
    for item in [30_000, 10_000, 6_500, 5_000, 4_000, 3_500]:
        xticks.append(_invi.photometry.magnitudes.Teff_to_BPRP(item))
        xticklabels.append(str(item))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    #Left y axis
    ax.set_ylabel("$g$ [mag]")
    ax.set_ylim(y_limit)


def HR(gc_gaia, sample_phot_red, sel, s=1.2):
    fig, ax = _fnc.plot.figure(1, 1, fc=(2.4,2.0))

    ax.scatter(gc_gaia['photometry_red']['bprp_red'],
               gc_gaia['photometry_red']['g_red'],
               s=s, c="k", linewidth=0.5, label="GDR3 - Red. Sim.")

    ax.scatter(sample_phot_red['bprp_red'][sel],
               sample_phot_red['g_red'][sel],
               s=s, c="r", linewidth=0.5, label="Mock")

    ax.legend(markerscale=3)

    x_limit = [-1.0, 3.0]
    y_limit = [22.5, 10.0]
    _set_axis(ax, 21.0, x_limit, y_limit)

#-----------------------------------------------------------------------------

def set_axes_sky_coord(ax, prm_gc):
    #Globular cluster
    ax.scatter(prm_gc['ICRS']['alpha'], prm_gc['ICRS']['delta'], s=1.5, c='r')

    dec_cut = prm_gc['selection']['dec_cut']
    ra_cut = prm_gc['selection']['ra_cut']
    kwargs = {'linestyle': '--', 'color': 'r', 'alpha': 0.5}
    ax.plot([0.0, 360.0], [dec_cut, dec_cut], **kwargs)
    ax.plot([ra_cut, ra_cut], [-90.0, 90.0], **kwargs)

    ax.legend(loc='lower right', markerscale=3.5)
    ax.set_aspect(1)
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_xlim(160.0, 330.0)
    ax.set_ylim(-70.0, 80.0)

#-----------------------------------------------------------------------------

