"""Plots for Jupyter notebooks in code/3-petar folder."""

import fnc as _fnc
import invi as _invi

__all__ = ["FSR", "ICRS"]

#-----------------------------------------------------------------------------
#3-results

def FSR(orb, orb_frw, s_dict):
    #-------------------------------------------------------
    def plots_orb(ax, x_orb, y_orb, x_orb_frw, y_orb_frw, x_label, y_label):
        ax.plot(x_orb, y_orb, color="k", linestyle="-", zorder=0)
        ax.plot(x_orb_frw, y_orb_frw, color="k", linestyle="--", zorder=0)
        ax.scatter(x_orb[0], y_orb[0], s=25.0, c="b", zorder=2)
        ax.scatter(x_orb[-1], y_orb[-1], s=25.0, c="g", zorder=2)
        ax.set_aspect(1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def plots_str(ax, x_str, y_str):
        ax.scatter(x_str, y_str, color="r", s=0.1, zorder=1)
    #-------------------------------------------------------
    _fig, ax = _fnc.plot.figure(2, 2, fc=(2.5,2.5))
    ax1, ax2, ax3, ax4 = ax[0,0], ax[0,1], ax[1,0], ax[1,1]
    car_orb = orb['orbit']['FSR']['car']
    cyl_orb = orb['orbit']['FSR']['cyl']

    car_orb_frw = orb_frw['orbit']['FSR']['car']
    cyl_orb_frw = orb_frw['orbit']['FSR']['cyl']

    car_str = s_dict['FSR']['car']
    cyl_str = s_dict['FSR']['cyl']

    plots_orb(ax1, car_orb['x'], car_orb['y'], car_orb_frw['x'], car_orb_frw['y'], "$x$ [kpc]", "$y$ [kpc]")
    plots_orb(ax2, cyl_orb['R'], cyl_orb['z'], cyl_orb_frw['R'], cyl_orb_frw['z'], "$R$ [kpc]", "$z$ [kpc]")
    plots_orb(ax3, car_orb['x'], car_orb['z'], car_orb_frw['x'], car_orb_frw['z'], "$x$ [kpc]", "$z$ [kpc]")
    plots_orb(ax4, car_orb['v_y'], car_orb['v_z'], car_orb_frw['v_y'], car_orb_frw['v_z'], "$v_y$ [kpc/Myr]", "$v_z$ [kpc/Myr]")

    plots_str(ax1, car_str['x'], car_str['y'])
    plots_str(ax2, cyl_str['R'], cyl_str['z'])
    plots_str(ax3, car_str['x'], car_str['z'])
    plots_str(ax4, car_str['v_y'], car_str['v_z'])


def ICRS(orb, orb_frw, s_dict):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(2.25,2.25))

    orb_sph = orb['orbit']['ICRS']
    orb_frw_sph = orb_frw['orbit']['ICRS']
    str_sph = s_dict['ICRS']

    ax.plot(orb_sph['alpha'], orb_sph['delta'], c="k", linewidth=2.0, linestyle="-", zorder=0)
    ax.plot(orb_frw_sph['alpha'], orb_frw_sph['delta'], c="k", linewidth=2.0, linestyle="--", zorder=0)

    ax.scatter(str_sph['alpha'], str_sph['delta'], c="r", s=0.1, zorder=1)

    ax.scatter(orb_sph['alpha'][0], orb_sph['delta'][0], s=25.0, c="b", zorder=2)
    ax.scatter(orb_sph['alpha'][-1], orb_sph['delta'][-1], s=25.0, c="g", zorder=2)

    _invi.plot.lines_b15(ax)

    ax.set_aspect(1)
    ax.set_xlim(160.0, 320.0)
    ax.set_ylim(-80.0, 80.0)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")

#-----------------------------------------------------------------------------
