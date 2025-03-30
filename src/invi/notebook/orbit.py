"""Plots for Jupyter notebooks in code/2-orbit folder."""

import fnc as _fnc
import invi as _invi

__all__ = ["orbit",
           "FSR", "ICRS"]

#-----------------------------------------------------------------------------
#1-orbit_M68

def FSR(orb):
    #----------------------------------------------------
    def plots(ax, x, y, x_label, y_label):
        ax.plot(x, y, color="k", linestyle="-", zorder=0)
        ax.scatter(x[0], y[0], s=25.0, c="r", zorder=1)
        ax.scatter(x[-1], y[-1], s=25.0, c="g", zorder=1)
        ax.set_aspect(1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    #----------------------------------------------------
    _fig, ax = _fnc.plot.figure(2, 2, fc=(2.5,2.5))
    ax1, ax2, ax3, ax4 = ax[0,0], ax[0,1], ax[1,0], ax[1,1]
    car = orb['orbit']['FSR']['car']
    cyl = orb['orbit']['FSR']['cyl']

    plots(ax1, car['x'], car['y'], "$x$ [kpc]", "$y$ [kpc]")
    plots(ax2, cyl['R'], cyl['z'], "$R$ [kpc]", "$z$ [kpc]")
    plots(ax3, car['x'], car['z'], "$x$ [kpc]", "$z$ [kpc]")
    plots(ax4, car['v_y'], car['v_z'], "$v_y$ [kpc/Myr]", "$v_z$ [kpc/Myr]")


def ICRS(orb):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(1.75,1.75))

    sph = orb['orbit']['ICRS']

    ax.plot(sph['alpha'], sph['delta'], c="k", linewidth=2.0, linestyle="-", zorder=0)

    ax.scatter(sph['alpha'][0], sph['delta'][0], s=25.0, c="r", zorder=1)
    ax.scatter(sph['alpha'][-1], sph['delta'][-1], s=25.0, c="g", zorder=1)

    _invi.plot.lines_b15(ax)

    ax.set_aspect(1)
    ax.set_xlim(160.0, 320.0)
    ax.set_ylim(-60.0, 80.0)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")

#-----------------------------------------------------------------------------
#2-long_short_axis_tube

def orbit(orb, prm):
    #---------------------------
    def set_axis(ax, x_label):
        ax.set_aspect(1)
        ax.set_xlabel( x_label)
        ax.set_ylabel("z [kpc]")
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(-30.0, 30.0)
    #---------------------------
    R = orb['orbit']['FSR']['cyl']['R']
    z = orb['orbit']['FSR']['cyl']['z']
    r = orb['orbit']['FSR']['sph']['r']

    q = prm['mw']['halo']['q']

    fig, ax = _fnc.plot.figure(1, 2, fc=(2,1.5))
    fig.suptitle(f"q = {q}")
    kwargs = {'alpha': 1.0, 'linewidth': 0.025}
    ax1, ax2 = ax[0], ax[1]

    ax1.plot(R, z, **kwargs)
    set_axis(ax1, "R [kpc]")

    ax2.plot(r, z, **kwargs)
    set_axis(ax2, "r [kpc]")

#-----------------------------------------------------------------------------
