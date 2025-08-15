"""Functions to cross match two star catalogues."""

import numpy as _np
import tqdm.auto as _tqdm

import invi as _invi

__all__ = ["distance", "id", "get_data"]

#-----------------------------------------------------------------------------

def distance(x_0, y_0, x_1, y_1, eps, metric='euclidean', verbose=True):
    """Cross match using the distance between two sources."""
    match metric:
        case 'euclidean':
            dist = _invi.misc.euclidean_distance
        case 'angular':
            dist = _invi.misc.angular_distance
        case _:
            raise ValueError("Allowed metrics: 'euclidean', 'angular'")

    n0 = len(x_0)
    n1 = len(x_1)

    matches = []
    indices = _np.arange(0, n1, 1)

    for i in _tqdm.tqdm(range(n0), disable=not verbose):
        d = dist(x_0[i], y_0[i], x_1, y_1)
        sel = d < eps
        idx = indices[sel]
        matches.append(idx)

        if verbose & (len(idx) > 1):
            print(f"[Warning] {i} matches with {idx} with minimum distance = {min(d[sel]):0.15E}")

    return matches

#-----------------------------------------------------------------------------

def id(id_0, id_1, verbose=True):
    """Cross match using the id of the sources."""
    n0 = len(id_0)
    n1 = len(id_1)

    matches = []
    indices = _np.arange(0, n1, 1)

    for i in _tqdm.tqdm(range(n0), disable=not verbose):
        idx = indices[id_0[i] == id_1]
        matches.append(idx)

        if verbose & (len(idx) > 1):
            print(f"[Warning] {i} matches with {idx}")

    return matches

#-----------------------------------------------------------------------------

def get_data(data_1, matches):
    """Get data from matches.

    Note
    ----
    1)  It takes the first element in case of multiple matches.
    2)  If 'data_1' are integers, this function converts them into floats
        because np.nan is only defined for floats."""
    data_0 = [_np.nan]*len(matches)

    for i, item in enumerate(matches):
        if len(item) > 0:
            data_0[i] = data_1[item[0]]

    return _np.asarray(data_0)

#-----------------------------------------------------------------------------
