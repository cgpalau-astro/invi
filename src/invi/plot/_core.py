"""General functions for plots."""

import numpy as _np
import tqdm as _tqdm

import fnc as _fnc
_pl = _fnc.utils.lazy.Import("pathlib")
_shutil = _fnc.utils.lazy.Import("shutil")
_mpl = _fnc.utils.lazy.Import("matplotlib")
_plt = _fnc.utils.lazy.Import("matplotlib.pyplot")
_sklearn = _fnc.utils.lazy.Import("sklearn")
_shapely = _fnc.utils.lazy.Import('shapely')

__all__ = ["save_figure", "lines_b15", "mixed_scatter", "hist_area", "color_line", "polygon"]

#-----------------------------------------------------------------------------

def save_figure(directory, name, fast=False):
    """Save figures for papers."""
    if not fast:
        _plt.savefig(f"{name}_300.png", dpi=300)
        _plt.savefig(f"{name}_600.png", dpi=600)
        _shutil.copy2(f"{name}_300.png", f"{directory}png_300/{name}_300.png")
        _shutil.copy2(f"{name}_600.png", f"{directory}png_600/{name}_600.png")
    else:
        _plt.savefig(f"{name}_300.png", dpi=300)
        _shutil.copy2(f"{name}_300.png", f"{directory}png_300/{name}_300.png")

#-----------------------------------------------------------------------------

def lines_b15(ax, zorder=0):
    """Plot limits of the disc at b=±15 deg and Galactic centre in ICRS."""
    base_path = _pl.Path(__file__).parent
    file_path = (base_path / "ICRS_b_15deg.csv.gz").resolve()

    kwargs = {'color': "k", 'alpha': 0.5, 'zorder': zorder}

    #Disc b=±15 deg lines
    alpha_0, delta_0, alpha_1, delta_1 = _np.loadtxt(file_path, delimiter=',', skiprows=4, unpack=True)
    ax.plot(alpha_0, delta_0, linestyle='--', linewidth=1.0, **kwargs)
    ax.plot(alpha_1, delta_1, linestyle='--', linewidth=1.0, **kwargs)

    #Galactic centre Sgr A*
    ax.scatter(266.4167, -29.0078, s=25.0, marker='x', linewidth=0.75, **kwargs)

#-----------------------------------------------------------------------------

def mixed_scatter(ax, a, b, color_a, color_b, **kargs):
    """Scatter plot where the points 'a' and 'b' are plotted mixed.

    Example
    -------
    import invi
    import fnc
    import scipy

    cov = [[1.0, 0.0], [0.0, 1.0]]
    norm_a = scipy.stats.multivariate_normal(mean=[1.0, 1.0], cov=cov)
    norm_b = scipy.stats.multivariate_normal(mean=[1.75, 1.75], cov=cov)

    size = 2_000
    a = norm_a.rvs(size=size, random_state=123).T
    b = norm_b.rvs(size=size, random_state=124).T

    fig, ax = fnc.plot.figure(1, 2, fc=(2,1))
    s = 5.0
    ax[0].scatter(a[0], a[1], s=s, c="k")
    ax[0].scatter(b[0], b[1], s=s, c="r")

    invi.plots.general.mixed_scatter(ax[1], a, b, 'k', 'r', s=s)"""
    #-------------------------------------------------------
    def plot(n, ax, a, b, color_a, color_b, **kargs):
        for i in _tqdm.tqdm(range(n), ncols=78):
            ax.scatter(a[0][i], a[1][i], c=color_a, **kargs)
            ax.scatter(b[0][i], b[1][i], c=color_b, **kargs)
    #-------------------------------------------------------
    #Shuffle points
    a[0], a[1] = _sklearn.utils.shuffle(a[0], a[1], random_state=123)
    b[0], b[1] = _sklearn.utils.shuffle(b[0], b[1], random_state=123)

    len_a = len(a[0])
    len_b = len(b[0])

    if len_a == len_b:
        plot(len_a, ax, a, b, color_a, color_b, **kargs)
    elif len_a > len_b:
        ax.scatter(a[0][len_b:], a[1][len_b:], c=color_a, **kargs)
        plot(len_b, ax, a, b, color_a, color_b, **kargs)
    else:
        mixed_scatter(ax, b, a, color_b, color_a, **kargs)

#-----------------------------------------------------------------------------

def hist_area(ax, x, bins, range, area, **kwargs):
    """Plot histogram with specified area.

    Example
    -------
    import numpy as np
    import fnc
    import invi

    x = [1, 2, 3, 4, 5, 5, 6]

    bins = 20
    rng = [0, 10]

    fig, ax = fnc.plot.figure(1,1, fc=(1.3,1.3))
    h = ax.hist(x, bins=bins, range=rng, density=True)
    invi.plots.general.hist_area(ax, x, bins=bins, range=rng, area=0.8, color="r")"""
    hist = _np.histogram(x, bins=bins, range=range, density=True)
    steps = hist[1][1:len(hist[1])]
    normalized_counts = hist[0]
    ax.step(steps, normalized_counts*area, **kwargs)

#-----------------------------------------------------------------------------

def color_line(ax, x, y, z=None, cmap=None, norm=None, linewidth=3.0, alpha=1.0):
    """Plot a colored line with coordinates x and y. Optionally specify colors
    in the array z, colormap, norm function and a line width.

    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt
    import fnc
    import invi

    phi = np.linspace(0.0, 2.0*np.pi, 1_000)
    x = np.sin(phi)
    y = np.cos(phi)

    fig, ax = fnc.plot.figure()
    cb = invi.plots.general.color_line(ax, x, y, z=phi, cmap='turbo', norm=plt.Normalize(0.0, 2.0*np.pi), linewidth=10, alpha=1.0)
    cbar = fig.colorbar(cb, label="$(0, 2\\pi)$", fraction=0.1, pad=0.05, extend=None)"""
    #-----------------------------------------------------------
    def make_segments(x, y, con=1):
        """Create list of line segments from x and y coordinates, in the
        format for _mpl.collections.LineCollection: an array of the form:
        numlines x (points per line) x 2 (x and y) array"""
        points = _np.array([x, y]).T.reshape(-1, 1, 2)
        if con == 1:
            segments = _np.concatenate([points[:-1], points[1:]], axis=1)
        else:
            segments = _np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)
        return segments
    #-----------------------------------------------------------
    #Default colors equally spaced on [0,1]:
    if z is None:
        z = _np.linspace(0.0, 1.0, len(x))

    #Special case if a single number:
    if not hasattr(z, "__iter__"):  #Check for numerical input
        z = _np.array([z])

    z = _np.asarray(z)

    if cmap is None:
        cmap = _plt.get_cmap('copper')

    if norm is None:
        norm = _plt.Normalize(0.0, 1.0)
    #-----------------------------------------------------------
    kwargs = {'array': z, 'cmap': cmap, 'norm': norm, 'linewidth': linewidth, 'alpha': alpha}

    segments1 = make_segments(x, y, 1)
    lc1 = _mpl.collections.LineCollection(segments1, **kwargs)

    segments2 = make_segments(x, y, 2)
    lc2 = _mpl.collections.LineCollection(segments2, **kwargs)

    ax.plot(x, y, alpha=0.0)
    ax.add_collection(lc1)
    ax.add_collection(lc2)

    return lc1

#-----------------------------------------------------------------------------

def polygon(ax, points_polygon, **kwargs):
    """Plot polygon."""

    polygon = _shapely.geometry.Polygon(shell=points_polygon)

    path = _mpl.path.Path.make_compound_path(_mpl.path.Path(_np.asarray(polygon.exterior.coords)[:, :2]),
                                             *[_mpl.path.Path(_np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors])

    patch = _mpl.patches.PathPatch(path, **kwargs)
    collection = _mpl.collections.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()

#-----------------------------------------------------------------------------
