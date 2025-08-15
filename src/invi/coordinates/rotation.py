"""Rotation along cartesian axes.

Note
----
1)  Angles in rad."""

import numpy as _np

__all__ = ["rotation_x", "rotation_y", "rotation_z",
           "xyz", "w", "aaf",
           "xyz_inv", "w_inv", "aaf_inv",
           "angles_to_matrix", "matrix_to_angles", "hessian_to_matrix"]

#-----------------------------------------------------------------------------

def rotation_x(x, y, z, alpha):
    """Rotation an angle alpha along the cartesian x axis."""
    a = x
    b = _np.cos(alpha)*y - _np.sin(alpha)*z
    c = _np.sin(alpha)*y + _np.cos(alpha)*z
    return a, b, c


def rotation_y(x, y, z, alpha):
    """Rotation an angle alpha along the cartesian y axis."""
    a =  _np.cos(alpha)*x + _np.sin(alpha)*z
    b =  y
    c = -_np.sin(alpha)*x + _np.cos(alpha)*z
    return a, b, c


def rotation_z(x, y, z, alpha):
    """Rotation an angle alpha along the cartesian z axis."""
    a = _np.cos(alpha)*x - _np.sin(alpha)*y
    b = _np.sin(alpha)*x + _np.cos(alpha)*y
    c = z
    return a, b, c

#-----------------------------------------------------------------------------

def xyz(x, y, z, alpha_x, alpha_y, alpha_z):
    """Rotation an angle alpha_i along the cartesian axes in the order z, y, x."""
    a, b, c = rotation_z(x, y, z, alpha_z)
    d, e, f = rotation_y(a, b, c, alpha_y)
    g, h, i = rotation_x(d, e, f, alpha_x)
    return g, h, i


def w(w_car, alpha_x, alpha_y, alpha_z):
    """Rotation an angle alpha_i along the cartesian axes in the order z, y, x."""
    x, y, z, vx, vy, vz = w_car
    a, b, c = xyz(x, y, z, alpha_x, alpha_y, alpha_z)
    va, vb, vc = xyz(vx, vy, vz, alpha_x, alpha_y, alpha_z)
    return _np.array([a, b, c, va, vb, vc])


def aaf(aaf, alpha_x, alpha_y, alpha_z):
    """Rotation an angle alpha_i along the cartesian axes in the order z, y, x."""
    a1, a2, a3 = xyz(aaf[0], aaf[1], aaf[2], alpha_x, alpha_y, alpha_z)
    j1, j2, j3 = xyz(aaf[3], aaf[4], aaf[5], alpha_x, alpha_y, alpha_z)
    f1, f2, f3 = xyz(aaf[6], aaf[7], aaf[8], alpha_x, alpha_y, alpha_z)
    return _np.array([a1, a2, a3, j1, j2, j3, f1, f2, f3])

#-----------------------------------------------------------------------------

def xyz_inv(x, y, z, alpha_x, alpha_y, alpha_z):
    """Rotation an angle alpha_i along the cartesian axes in the order x, y, z."""
    a, b, c = rotation_x(x, y, z, alpha_x)
    d, e, f = rotation_y(a, b, c, alpha_y)
    g, h, i = rotation_z(d, e, f, alpha_z)
    return g, h, i


def w_inv(w_car, alpha_x, alpha_y, alpha_z):
    """Rotation an angle -alpha_i along the cartesian axes in the order x, y, z."""
    x, y, z, vx, vy, vz = w_car
    a, b, c = xyz_inv(x, y, z, -alpha_x, -alpha_y, -alpha_z)
    va, vb, vc = xyz_inv(vx, vy, vz, -alpha_x, -alpha_y, -alpha_z)
    return _np.array([a, b, c, va, vb, vc])


def aaf_inv(aaf, alpha_x, alpha_y, alpha_z):
    """Rotation an angle -alpha_i along the cartesian axes in the order x, y, z."""
    a1, a2, a3 = xyz_inv(aaf[0], aaf[1], aaf[2], -alpha_x, -alpha_y, -alpha_z)
    j1, j2, j3 = xyz_inv(aaf[3], aaf[4], aaf[5], -alpha_x, -alpha_y, -alpha_z)
    f1, f2, f3 = xyz_inv(aaf[6], aaf[7], aaf[8], -alpha_x, -alpha_y, -alpha_z)
    return _np.array([a1, a2, a3, j1, j2, j3, f1, f2, f3])

#-----------------------------------------------------------------------------

def angles_to_matrix(alpha_x, alpha_y, alpha_z, order='xyz', verbose=True):
    """Return and print the rotation matrix R=Rx·Ry·Rz='xyz'=order given the
    rotation angles alpha_i in rad."""

    Rx = _np.array([[1.0, 0.0, 0.0],
                    [0.0, _np.cos(alpha_x), -_np.sin(alpha_x)],
                    [0.0, _np.sin(alpha_x), _np.cos(alpha_x)]])

    Ry = _np.array([[_np.cos(alpha_y), 0.0, _np.sin(alpha_y)],
                    [0.0, 1.0, 0.0],
                    [-_np.sin(alpha_y), 0.0, _np.cos(alpha_y)]])

    Rz = _np.array([[_np.cos(alpha_z), -_np.sin(alpha_z), 0.0],
                    [_np.sin(alpha_z), _np.cos(alpha_z), 0.0],
                    [0.0, 0.0, 1.0]])

    Ri = {'x': Rx, 'y': Ry, 'z': Rz}

    R = _np.dot(Ri[order[0]], _np.dot(Ri[order[1]], Ri[order[2]]))

    if verbose:
        print(f"np.array({_np.array_str(R, precision=3)})")

    return R


def matrix_to_angles(R):
    """Return the rotation angles in rad assuming the matrix R=Rx·Ry·Rz='xyz'=order.

    Note
    ----
    1)  This is a particular case, see: https://www.geometrictools.com/Documentation/EulerAngles.pdf"""

    varphi_x = _np.arctan2(-R[1][2], R[2][2])
    varphi_y = _np.arcsin(R[0][2])
    varphi_z = _np.arctan2(-R[0][1], R[0][0])

    return _np.array([varphi_x, varphi_y, varphi_z])


def hessian_to_matrix(H, verbose=True):
    """Compute the Rotation matrix given the Hessian matrix.

    Note
    ----
    1)  The eigenvalues are obtained by changing the basis of the Hessian
        matrix (such as a mix tensor):
        eigval = np.dot(np.linalg.inv(S), np.dot(H, S)).diagonal()"""

    #Eigenvalues and eigenvectors of the Hessian matrix (eigenvectors are the columns)
    eigval, eigvec = _np.linalg.eig(H)

    #Index sorted from major to minor eigenvalue
    idx = _np.flip(_np.argsort(_np.abs(eigval)))

    #Reorder the eigenvectors (columns)
    S = eigvec[:, [idx]].reshape(3,3)

    #The rotation matrix is the inverse of the eigvectors
    R = _np.linalg.inv(S)

    if verbose:
        print(f"np.array({_np.array_str(R, precision=3)})")

    return R

#-----------------------------------------------------------------------------
