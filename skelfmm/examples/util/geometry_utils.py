"""
Geometry helpers for the example scripts.

The sphere-panel utilities here are lightly inspired by the geometry helpers
used in FLAM's example problems, but the implementations in this file are local
to this repo.
"""

import numpy as np
from scipy.spatial import ConvexHull

from simpletree import get_point_dist
from simpletree.built_in_points import get_sphere_surface


POINT_GEOMETRIES = (
    "square",
    "cube",
    "circle_surface",
    "cube_surface",
    "sphere_surface",
    "annulus",
    "curvy_annulus",
)


def make_point_geometry(geometry: str, n_points: int) -> np.ndarray:
    """Return an `N x d` point cloud for one of the built-in demo geometries."""
    if geometry not in POINT_GEOMETRIES:
        available = ", ".join(POINT_GEOMETRIES)
        raise ValueError(f"Unknown geometry {geometry!r}. Available geometries: {available}.")
    return get_point_dist(n_points, geometry)

def _icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=float,
    )
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )
    return V, F


def _subdivide_icosphere(V, F):
    midpoint_cache = {}
    verts = V.tolist()

    def midpoint(i, j):
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        p = 0.5 * (np.array(verts[i]) + np.array(verts[j]))
        p /= np.linalg.norm(p)
        idx = len(verts)
        verts.append(p.tolist())
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for tri in F:
        i, j, k = map(int, tri)
        a = midpoint(i, j)
        b = midpoint(j, k)
        c = midpoint(k, i)
        new_faces.extend(
            [
                [i, a, c],
                [j, b, a],
                [k, c, b],
                [a, b, c],
            ]
        )

    return np.array(verts, dtype=float), np.array(new_faces, dtype=int)


def make_icosphere_mesh(n_panels=5120, radius=1.0):
    """
    Generate an icosphere with exactly ``20 * 4**k`` triangular panels.

    This matches the classical sphere-panel hierarchy used in the older BIE
    examples much more closely than the point-sample convex-hull mesh.
    """
    if n_panels < 20 or n_panels % 20 != 0:
        raise ValueError("n_panels must be of the form 20 * 4**k")

    ratio = n_panels // 20
    nsub = 0
    while ratio > 1 and ratio % 4 == 0:
        ratio //= 4
        nsub += 1
    if ratio != 1:
        raise ValueError("n_panels must be of the form 20 * 4**k")

    V, F = _icosahedron()
    for _ in range(nsub):
        V, F = _subdivide_icosphere(V, F)
    return radius * V, F


def make_sphere_mesh(n_points=128, radius=1.0):
    """
    Generate a simple triangular sphere mesh with `V` as `N x 3` and `F` as `M x 3`.

    Attribution: the example-side sphere geometry workflow is inspired by the
    geometry construction utilities distributed with FLAM.
    """
    verts = get_sphere_surface(np.zeros(3), 2.0 * radius, n_points)
    hull = ConvexHull(verts)
    V = verts.copy()
    F = hull.simplices.copy()

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    flip = np.sum(normals * centroids, axis=1) < 0
    tmp = F[flip, 1].copy()
    F[flip, 1] = F[flip, 2]
    F[flip, 2] = tmp
    return V, F


def triangle_panel_geometry(V, F):
    """Return panel centers, outward normals, and areas in row-major `N x 3` form."""
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(face_cross, axis=1)

    centers = (v0 + v1 + v2) / 3.0
    normals = face_cross / np.linalg.norm(face_cross, axis=1, keepdims=True)
    return centers, normals, area
