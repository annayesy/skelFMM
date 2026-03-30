"""
Triangle quadrature helpers for the example scripts.

These routines are inspired by the geometry/quadrature utilities used in FLAM's
example problems, but the implementations in this file are local to this repo.
"""

import numpy as np


def gqgw(alpha, beta, mu):
    """Golub-Welsch quadrature from a three-term recurrence."""
    T = np.diag(beta, -1) + np.diag(alpha) + np.diag(beta, 1)
    eigvals, eigvecs = np.linalg.eig(T)
    order = np.argsort(eigvals)
    x = np.real_if_close(eigvals[order])
    w = np.real_if_close(mu * eigvecs[0, order] ** 2)
    return x, w


def glegquad(n, a=-1.0, b=1.0):
    """Gauss-Legendre quadrature on [a, b]."""
    if n < 1:
        raise ValueError("Quadrature order must be at least 1")

    alpha = np.zeros(n)
    beta = 0.5 / np.sqrt(1 - (2 * np.arange(1, n, dtype=float)) ** (-2))
    x, w = gqgw(alpha, beta, 2.0)
    x = 0.5 * ((b - a) * x + a + b)
    w = 0.5 * (b - a) * w
    return x, w


def tri3transrot(V, F):
    """
    Translate/rotate each 3D triangle into a 2D reference frame.

    Parameters
    ----------
    V : ndarray, shape (nverts, 3)
        Vertex coordinates in row-major form.
    F : ndarray, shape (ntri, 3)
        Triangle indices in row-major form.

    Returns
    -------
    trans : ndarray, shape (ntri, 3)
        Translation vectors that move each triangle's first vertex to the origin.
    rot : ndarray, shape (ntri, 3, 3)
        Per-triangle rotation matrices whose rows form the local orthonormal frame.
    v2 : ndarray, shape (ntri,)
        Length of the first triangle edge in local coordinates.
    v3 : ndarray, shape (ntri, 2)
        Coordinates of the third vertex in the local 2D frame.
    """
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    trans = -v0
    edge10 = v1 - v0
    edge20 = v2 - v0

    edge10_norm = np.linalg.norm(edge10, axis=1)
    ntri = F.shape[0]
    rot = np.zeros((ntri, 3, 3))
    rot[:, 0, :] = edge10 / edge10_norm[:, None]

    normals = np.cross(edge10, edge20)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rot[:, 2, :] = normals
    rot[:, 1, :] = np.cross(rot[:, 2, :], rot[:, 0, :])

    v3_local = np.zeros((ntri, 2))
    for i in range(ntri):
        v3_local[i] = rot[i, :2, :] @ edge20[i]
    return trans, rot, edge10_norm, v3_local


def qmap_sqtri2(x0, y0, w0, v2, v3):
    """
    Map unit-square quadrature to a 2D reference triangle.

    Parameters
    ----------
    x0, y0, w0 : ndarray, shape (nq,)
        Unit-square quadrature nodes and weights.
    v2 : float
        x-coordinate of the second triangle vertex in the local 2D frame.
    v3 : ndarray, shape (2,)
        Coordinates of the third triangle vertex in the local 2D frame.
    """
    y = x0 * y0
    w = w0 * x0

    a0 = np.array([v2, 0.0])
    A = np.column_stack((a0, np.asarray(v3) - a0))

    z = A @ np.vstack([x0, y])
    x = z[0]
    y = z[1]
    w = w * np.linalg.det(A)
    return x, y, w
