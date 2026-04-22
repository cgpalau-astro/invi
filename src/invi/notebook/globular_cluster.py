"""Plots for Jupyter notebooks in code/1-globular_cluster folder."""

import numpy as _np

import fnc as _fnc
import invi as _invi
import invi.units as _un
import invi.photometry.magnitudes as _mg

__all__ = ["plummer_comparison", "plummer_log_comparison",
           "king_comparison", "king",
           "ra_parallax", "sky_coord", "pm", "HR_gaia", "parallax",
           "outliers", "HR", "mass"]

#-----------------------------------------------------------------------------

def _set_axis(ax, distance, x_limit, y_limit):
    #Plot horizontal line marking the Gaia magnitude limit
    gaia_limit = _mg.m_to_M(20.5, distance)
    ax.plot(x_limit, [gaia_limit, gaia_limit], linestyle="--", color='0.7')
    #--------------------------------------------------------
    #Lower x axis
    ax.set_xlabel("$(G_{\\rm BP}-G_{\\rm RP})$ [mag]")
    ax.set_xlim(x_limit)
    #--------------------------------------------------------
    #Upper x axis
    ax2 = ax.twiny()
    ax2.set_xlabel("$T_{\\rm eff}$ [K]")
    ax2.set_xlim(x_limit)

    xticks = []
    xticklabels = []
    for item in [30_000, 10_000, 6_500, 5_000, 4_000, 3_500]:
        xticks.append(_mg.Teff_to_BPRP(item))
        xticklabels.append(str(item))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    #--------------------------------------------------------
    #Left y axis
    ax.set_ylabel("$G$ [mag]")
    ax.set_ylim(y_limit)
    #--------------------------------------------------------
    #Right y axis
    ax2 = ax.twinx()
    ax2.set_ylabel("$g$ [mag]")
    ax2.set_ylim(_mg.M_to_m(y_limit[0], distance),
                 _mg.M_to_m(y_limit[1], distance))

    yticks = []
    yticklabels = []
    for item in range(11, 27+2, 2):
        yticks.append(item)
        yticklabels.append(str(item))

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels)
    #--------------------------------------------------------

#-----------------------------------------------------------------------------
#1-Plummer

def _radial_distribution(nbody):
    x = nbody[0]
    y = nbody[1]
    z = nbody[2]
    r = _np.sqrt(x**2.0 + y**2.0 + z**2.0)

    vx = nbody[3]
    vy = nbody[4]
    vz = nbody[5]
    mod_v = _np.sqrt(vx**2.0 + vy**2.0 + vz**2.0)

    return r, mod_v


def plummer_comparison(nbody, label):
    """Comparison r and |v| between two methods."""
    nbody[0] = _un.galactic_to_petar(nbody[0])
    nbody[1] = _un.galactic_to_petar(nbody[1])

    r_0, mod_v_0 = _radial_distribution(nbody[0])
    r_1, mod_v_1 = _radial_distribution(nbody[1])

    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.5,1.0))
    kwargs = {'bins': 200, 'histtype': 'step', 'density': True}

    _h = ax[0].hist(r_0, range=[0.0, 50.0], label=label[0], **kwargs)
    _h = ax[0].hist(r_1, range=[0.0, 50.0], label=label[1], **kwargs)
    ax[0].set_xlabel("r [pc]")
    ax[0].set_xlim(0.0, 50.0)
    ax[0].legend()

    _h = ax[1].hist(mod_v_0, range=[0.0, 20.0], **kwargs)
    _h = ax[1].hist(mod_v_1, range=[0.0, 20.0], **kwargs)
    ax[1].set_xlabel("|v| [pc/Myr]")
    ax[1].set_xlim(0.0, 20.0)


def plummer_log_comparison(nbody, label):
    """Comparison r and |v| between two methods in log scale."""
    nbody[0] = _un.galactic_to_petar(nbody[0])
    nbody[1] = _un.galactic_to_petar(nbody[1])

    r_0, mod_v_0 = _radial_distribution(nbody[0])
    r_1, mod_v_1 = _radial_distribution(nbody[1])

    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.5,1.0))
    kwargs = {'bins': 200, 'histtype': 'step', 'density': True}

    _h = ax[0].hist(r_0, range=[30.0, 300.0], label=label[0], **kwargs)
    _h = ax[0].hist(r_1, range=[30.0, 300.0], label=label[1], **kwargs)
    ax[0].set_xlabel("r [pc]")
    ax[0].set_xlim(30.0, 300.0)
    ax[0].set_ylim(1E-7, 1E-1)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()

    _h = ax[1].hist(mod_v_0, range=[1.0, 30.0], **kwargs)
    _h = ax[1].hist(mod_v_1, range=[1.0, 30.0], **kwargs)
    ax[1].set_xlabel("|v| [pc/Myr]")
    ax[1].set_xlim(1.0, 30.0)
    ax[1].set_ylim(1E-5, 1E0)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

#-----------------------------------------------------------------------------
#2-agama_limepy_comparison

def king_comparison(nbody_agama, nbody_limepy):
    #-----------------------------------------------------------
    def radial_distribution(x):
        r = _np.sqrt(x[0]**2.0 + x[1]**2.0 + x[2]**2.0)
        modv = _np.sqrt(x[3]**2.0 + x[4]**2.0 + x[5]**2.0)
        return r, modv
    #-----------------------------------------------------------
    #[kpc, kpc/Myr] to [pc, pc/Myr]
    nbody_agama = _un.galactic_to_petar(nbody_agama)
    nbody_limepy = _un.galactic_to_petar(nbody_limepy)
    #-----------------------------------------------------------
    r_agama, modv_agama = radial_distribution(nbody_agama)
    r_limepy, modv_limepy = radial_distribution(nbody_limepy)
    #-----------------------------------------------------------
    _fig, ax = _fnc.plot.figure(2, 2, fc=(3.0, 3.0))
    ax1, ax2, ax3, ax4 = ax[0,0], ax[0,1], ax[1,0], ax[1,1]
    kwargs = {'bins': 1_000, 'density': False, 'histtype': 'step'}
    #-----------------------------------------------------------
    limit = 400.0
    ax1.hist(r_agama, range=[0.0, limit], label="Agama", **kwargs)
    ax1.hist(r_limepy, range=[0.0, limit], label="Limepy", **kwargs)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(2.0, limit)
    ax1.set_ylim(1.0E-1, 1.0E6)
    ax1.set_xlabel("r [pc]")
    #-----------------------------------------------------------
    limit = 20.0
    ax2.hist(r_agama, range=[0.0, limit], label="Agama", **kwargs)
    ax2.hist(r_limepy, range=[0.0, limit], label="Limepy", **kwargs)
    ax2.set_xlim(0.0, limit)
    ax2.set_ylim(0.0, 3.0E3)
    ax2.set_xlabel("r [pc]")
    ax2.legend()
    #-----------------------------------------------------------
    limit = 20.0
    ax3.hist(modv_agama, range=[0.0, limit], label="Agama", **kwargs)
    ax3.hist(modv_limepy, range=[0.0, limit], label="Limepy", **kwargs)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(0.01, limit)
    ax3.set_ylim(1.0E-1, 1.0E5)
    ax3.set_xlabel("|v| [pc/Myr]")
    #-----------------------------------------------------------
    limit = 20.0
    ax4.hist(modv_agama, range=[0.0, limit], label="Agama", **kwargs)
    ax4.hist(modv_limepy, range=[0.0, limit], label="Limepy", **kwargs)
    ax4.set_xlim(0.0, limit)
    ax4.set_ylim(0.0, 4.0E3)
    ax4.set_xlabel("|v| [pc/Myr]")
    ax4.legend()
    #-----------------------------------------------------------


def king(nbody):
    #-----------------------------------------------------------
    def radial_distribution(x):
        r = _np.sqrt(x[0]**2.0 + x[1]**2.0 + x[2]**2.0)
        modv = _np.sqrt(x[3]**2.0 + x[4]**2.0 + x[5]**2.0)
        return r, modv
    #-----------------------------------------------------------
    #[kpc, kpc/Myr] to [pc, pc/Myr]
    nbody = _un.galactic_to_petar(nbody)
    #-----------------------------------------------------------
    r, modv = radial_distribution(nbody)
    #-----------------------------------------------------------
    _fig, ax = _fnc.plot.figure(2, 2, fc=(3.0, 3.0))
    ax1, ax2, ax3, ax4 = ax[0,0], ax[0,1], ax[1,0], ax[1,1]
    kwargs = {'label': "Agama", 'bins': 1_000, 'density': False,
              'histtype': 'step'}
    #-----------------------------------------------------------
    limit = 400.0
    ax1.hist(r, range=[0.0, limit], **kwargs)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(2.0, limit)
    ax1.set_ylim(1.0E-1, 1.0E5)
    ax1.set_xlabel("r [pc]")
    #-----------------------------------------------------------
    limit = 20.0
    ax2.hist(r, range=[0.0, limit], **kwargs)
    ax2.set_xlim(0.0, limit)
    ax2.set_ylim(0.0, 1.5E3)
    ax2.set_xlabel("r [pc]")
    ax2.legend()
    #-----------------------------------------------------------
    limit = 20.0
    ax3.hist(modv, range=[0.0, limit], **kwargs)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(0.01, limit)
    ax3.set_ylim(1.0E-1, 1.0E4)
    ax3.set_xlabel("|v| [pc/Myr]")
    #-----------------------------------------------------------
    limit = 20.0
    ax4.hist(modv, range=[0.0, limit], **kwargs)
    ax4.set_xlim(0.0, limit)
    ax4.set_ylim(0.0, 2.0E3)
    ax4.set_xlabel("|v| [pc/Myr]")
    #-----------------------------------------------------------

#-----------------------------------------------------------------------------
#3-query_M68

def ra_parallax(gc_gaia, s=0.2):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(1.5,1.5))
    kwargs = {'s':s, 'c':"k", 'linewidth':0.5}
    ax.scatter(gc_gaia['ICRS']['alpha'], gc_gaia['parallax']['parallax'], **kwargs)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("parallax [mas]")


def sky_coord(gc_gaia, s=0.2):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(1.5,1.5))
    kwargs = {'s':s, 'c':"k", 'linewidth':0.5}
    ax.scatter(gc_gaia['ICRS']['alpha'], gc_gaia['ICRS']['delta'], **kwargs)
    ax.set_aspect(1)
    ax.set_xlabel("$\\alpha$ [deg]")
    ax.set_ylabel("$\\delta$ [deg]")


def pm(gc_gaia, s=0.2):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(1.5,1.5))
    kwargs = {'s':s, 'c':"k", 'linewidth':0.5}
    ax.scatter(gc_gaia['ICRS']['mu_alpha_str'], gc_gaia['ICRS']['mu_delta'], **kwargs)
    ax.set_aspect(1)
    ax.set_xlabel("$\\mu_{\\alpha*}$ [mas/yr]")
    ax.set_ylabel("$\\mu_{\\delta}$ [mas/yr]")


def HR_gaia(gc_gaia, prm, s=0.2):
    _fig, ax = _fnc.plot.figure(1, 1, fc=(2.0,2.0))
    kwargs = {'s':s, 'linewidth':0.5}

    ax.scatter(gc_gaia['photometry']['BPRP'],
               gc_gaia['photometry']['G'],
               label="Reddening corrected", c="k", **kwargs)
    ax.legend(markerscale=3)

    x_limit = [-0.5, 2.0]
    y_limit = [7.5, -6.0]
    distance = prm['M68']['ICRS']['r']
    _set_axis(ax, distance, x_limit, y_limit)


def parallax(gc_gaia, prm, bins=300, x_limit=None):
    parallax = gc_gaia['parallax']['parallax']
    parallax_zpc = gc_gaia['parallax']['parallax_corrected']
    parallax_gc = _invi.coordinates.r_to_parallax(prm['M68']['ICRS']['r'])

    if x_limit is None:
        x_limit = [_np.min(parallax), _np.max(parallax)]

    _fig, ax = _fnc.plot.figure(1, 1, fc=(2.5, 1.25))

    kwargs = {'bins':bins, 'histtype':"step", 'density':True, 'range':x_limit}

    h_plx = ax.hist(parallax, color="k", label="Parallax", **kwargs)
    y_limit_plx = _np.max(h_plx[0])*1.1

    h_zpc = ax.hist(parallax_zpc, color="r", label="Parallax zero point corrected", **kwargs)
    y_limit_zpc = _np.max(h_zpc[0])*1.1

    y_limit = _np.max([y_limit_plx, y_limit_zpc])

    #Parallax from the distance of the globular cluster
    ax.plot([parallax_gc, parallax_gc], [0.0, y_limit],
            linestyle="--", c="b",
            label=f"plx = {parallax_gc:0.3f} [mas] \n rh = {1.0/parallax_gc:0.3f} [kpc]")

    ax.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax.set_xlabel("parallax [mas]")
    ax.set_xlim(x_limit)
    ax.set_ylim(0.0, y_limit)

#-----------------------------------------------------------------------------
#4-synthetic_population

def outliers(gc_gaia, distance, s=1.5):
    fig, ax = _fnc.plot.figure(1, 1, fc=(2.0,2.0))
    fig.suptitle("HR-Diagram: No reddening correction")

    sel = gc_gaia['components']['gc']
    ax.scatter(gc_gaia['photometry']['BPRP'][sel],
               gc_gaia['photometry']['G'][sel],
               s=s, c="k", linewidth=0.5, label="Globular cluster")

    sel = gc_gaia['components']['outliers']
    ax.scatter(gc_gaia['photometry']['BPRP'][sel],
               gc_gaia['photometry']['G'][sel],
               s=s, c="r", linewidth=0.5, label="Outliers")

    ax.legend(markerscale=3)

    x_limit = [-0.5, 2.0]
    y_limit = [7.5, -6.0]
    _set_axis(ax, distance, x_limit, y_limit)


def HR(gc_gaia, distance, synthetic_population, s=1.2):
    fig, ax = _fnc.plot.figure(1, 1, fc=(2.4,2.0))
    sel = gc_gaia['components']['gc']

    ax.scatter(gc_gaia['photometry']['BPRP'][sel],
               gc_gaia['photometry']['G'][sel],
               s=s, c="k", linewidth=0.5, label="Reddening corrected")

    im = ax.scatter(synthetic_population['BPRP'],
                    synthetic_population['G'],
                    s=s, c=synthetic_population['mass'], linewidth=0.5,
                    label="Synthetic population", cmap='rainbow', vmin=0.0, vmax=0.8)
    ax.legend(markerscale=3)
    _cbar = fig.colorbar(im, label="Star mass [M$_{sun}$]")

    x_limit = [-1.0, 3.0]
    y_limit = [13.0, -5.5]
    _set_axis(ax, distance, x_limit, y_limit)


def mass(synthetic_population):
    _fig, ax = _fnc.plot.figure(1, 2, fc=(2.0,1.0))
    ax1, ax2 = ax[0], ax[1]

    mass = synthetic_population['mass']
    kwargs = {'bins': 200, 'histtype': "step", 'density': True}

    ax1.hist(mass, range=[0.0, 1.0], **kwargs)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("mass [M_sun]")

    ax2.hist(mass, range=[0.08, 1.0], **kwargs)

    kroupa = _invi.globular_cluster.synthetic_population.Kroupa()
    x = _np.linspace(0.1, 0.792, 1_000)
    y = kroupa.pdf(x)
    ax2.plot(x, y, color='r')

    ax2.set_xlim(0.08, 1.0)
    ax2.set_xlabel("mass [M_sun]")
    ax2.set_yscale('log')
    ax2.set_xscale('log')

#-----------------------------------------------------------------------------
