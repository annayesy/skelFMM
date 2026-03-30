from threadpoolctl import ThreadpoolController
import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.linalg import qr, solve_triangular
from scipy.special import j0, y0
from simpletree import get_cube_surface
from simpletree.built_in_points import get_sphere_surface

# Initialize a controller for managing thread pool settings
controller = ThreadpoolController()


def _skeleton_id_pivoted_qr(K_stacked, rank_or_eps):
    ncols = K_stacked.shape[1]
    R, piv = qr(
        K_stacked,
        mode="r",
        pivoting=True,
        overwrite_a=True,
        check_finite=False,
    )
    R = np.asarray(R)
    R = R[:ncols, :]

    if rank_or_eps < 1:
        diag_abs = np.abs(np.diag(R))
        if diag_abs.size == 0 or diag_abs[0] == 0:
            rank = 0
        else:
            cutoff = rank_or_eps * diag_abs[0]
            rank = int(np.count_nonzero(diag_abs > cutoff))
    else:
        rank = int(rank_or_eps)

    rank = max(0, min(rank, ncols))
    idx = np.asarray(piv, dtype=int)

    if rank == 0 or rank == ncols:
        proj = np.zeros((rank, ncols - rank), dtype=K_stacked.dtype)
    else:
        R11 = R[:rank, :rank]
        R12 = R[:rank, rank:]
        proj = solve_triangular(
            R11,
            R12,
            lower=False,
            overwrite_b=True,
            check_finite=False,
        )

    return rank, idx, proj


def _pairwise_diff_dist2(XX, YY):
    diff = XX[:, None, :] - YY[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return diff, dist2


def _pairwise_dist(XX, YY):
    assert XX.shape[-1] <= 3, "XX must have dimension <= 3"
    assert YY.shape[-1] <= 3, "YY must have dimension <= 3"
    return scipy_cdist(XX, YY, metric="euclidean")


def _sphere_proxy_normals(proxy_points, box_center):
    normals = proxy_points - np.asarray(box_center)[None, :]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norms


def _get_proxy_points(operator, box_center, box_length, nproxy):
    proxy_length = 2.95 * box_length
    if operator.proxy_surface == "cube":
        return get_cube_surface(np.asarray(box_center), proxy_length, nproxy=nproxy)
    if operator.proxy_surface == "sphere":
        return get_sphere_surface(np.asarray(box_center), proxy_length, npoints=nproxy)
    raise ValueError(f"Unsupported proxy surface {operator.proxy_surface!r}")


def _get_proxy_geometry(operator, proxy_points, box_center, side):
    if operator.geometry_side == side and operator.proxy_surface == "sphere":
        return _sphere_proxy_normals(proxy_points, box_center)
    return None


def build_stacked_proxy_matrix(operator, x_box, box_center, box_length, nproxy, *, geom=None, param=0.0):
    x_proxy = _get_proxy_points(operator, box_center, box_length, nproxy)

    if operator.is_symmetric and geom is None:
        K_BP = operator.numpy_matrix(
            x_box,
            x_proxy,
            param=param,
            assume_not_self_interaction=True,
        )
        target_shape = (2 * K_BP.shape[1], K_BP.shape[0])
        K_trans = K_BP.T
        K_stacked = np.vstack((K_trans, K_trans))
        assert K_stacked.shape == target_shape
        return K_stacked

    proxy_src_geom = _get_proxy_geometry(operator, x_proxy, box_center, "source")
    proxy_trg_geom = _get_proxy_geometry(operator, x_proxy, box_center, "target")
    K_BP = operator.numpy_matrix(
        x_box,
        x_proxy,
        geom_trg=geom,
        geom_src=proxy_src_geom,
        param=param,
        assume_not_self_interaction=True,
    )
    K_PB = operator.numpy_matrix(
        x_proxy,
        x_box,
        geom_trg=proxy_trg_geom,
        geom_src=geom,
        param=param,
        assume_not_self_interaction=True,
    )
    target_shape = (K_PB.shape[0] + K_BP.shape[1], K_PB.shape[1])
    K_stacked = np.vstack((K_PB, K_BP.T))
    assert K_stacked.shape == target_shape
    return K_stacked

######################################### KERNEL FUNCTIONS #########################################

def laplace_2d(XX, YY, param=0, assume_not_self_interaction=False):
    """
    Compute the 2D Laplace kernel function.
    
    Parameters:
    - XX: (N, 2) array-like, first set of points
    - YY: (M, 2) array-like, second set of points
    - param: Not used, kept for consistency
    
    Returns:
    - Logarithmic potential matrix of shape (N, M).
    """
    assert param == 0, "Parameter must be 0 for Laplace kernel"
    assert XX.shape[-1] == 2, "XX must have 2 dimensions"
    assert YY.shape[-1] == 2, "YY must have 2 dimensions"
    
    ZZ = _pairwise_dist(XX, YY)
    if not assume_not_self_interaction:
        zero_mask = ZZ == 0
        if np.any(zero_mask):
            ZZ[zero_mask] = 1.0
    return np.log(ZZ, out=ZZ)

def laplace_3d(XX, YY, param=0, assume_not_self_interaction=False):
    """
    Compute the 3D Laplace kernel function.
    
    Parameters:
    - XX: (N, 3) array-like, first set of points
    - YY: (M, 3) array-like, second set of points
    - param: Not used, kept for consistency
    
    Returns:
    - Reciprocal potential matrix of shape (N, M).
    """
    assert param == 0, "Parameter must be 0 for Laplace kernel"
    assert XX.shape[-1] == 3, "XX must have 3 dimensions"
    assert YY.shape[-1] == 3, "YY must have 3 dimensions"
    
    ZZ = _pairwise_dist(XX, YY)
    if assume_not_self_interaction:
        return np.reciprocal(ZZ, out=ZZ)

    zero_mask = ZZ == 0
    if np.any(zero_mask):
        ZZ[zero_mask] = np.inf
    return np.reciprocal(ZZ, out=ZZ)

def helmholtz_2d(XX, YY, param, assume_not_self_interaction=False):
    """
    Compute the 2D Helmholtz kernel function.
    
    Parameters:
    - XX: (N, 2) array-like, first set of points
    - YY: (M, 2) array-like, second set of points
    - param: Helmholtz parameter
    
    Returns:
    - Helmholtz potential matrix of shape (N, M).
    """
    ZZ = _pairwise_dist(XX, YY)
    zero_mask = None
    if not assume_not_self_interaction:
        zero_mask = ZZ == 0
        if np.any(zero_mask):
            ZZ[zero_mask] = 1.0
    ZZ *= param

    result = y0(ZZ)
    result = result.astype(np.cdouble)
    result *= 1j  # Complex term
    result += j0(ZZ, out=ZZ)
    if zero_mask is not None and np.any(zero_mask):
        result[zero_mask] = 0
    return result

def helmholtz_3d(XX, YY, param, assume_not_self_interaction=False):
    """
    Compute the 3D Helmholtz kernel function.
    
    Parameters:
    - XX: (N, 3) array-like, first set of points
    - YY: (M, 3) array-like, second set of points
    - param: Helmholtz parameter
    
    Returns:
    - Helmholtz potential matrix of shape (N, M).
    """
    ZZ = _pairwise_dist(XX, YY)
    zero_mask = None
    if not assume_not_self_interaction:
        zero_mask = ZZ == 0

    ZZ_double = ZZ.astype(np.cdouble)
    numerator = np.exp(+1j * param * ZZ_double, out=ZZ_double)

    if assume_not_self_interaction:
        return np.divide(numerator, ZZ, out=numerator)

    if np.any(zero_mask):
        ZZ[zero_mask] = np.inf
    result = np.divide(numerator, ZZ, out=numerator)
    if np.any(zero_mask):
        result[zero_mask] = 0
    return result

def laplace_3d_double_layer(XX, YY, param=0, src_geom=None, assume_not_self_interaction=False):
    r"""
    Compute the 3D Laplace double-layer kernel
    D(x,y) = \partial_{n_y} (1 / (4 pi |x-y|)).

    Parameters:
    - XX: (M, 3) target points
    - YY: (N, 3) source points
    - param: Not used, kept for API consistency
    - src_geom: (N, 3) source normals/dipole directions

    Returns:
    - Dense interaction matrix of shape (M, N).
    """
    assert param == 0, "Parameter must be 0 for Laplace double-layer kernel"
    assert XX.shape[-1] == 3, "XX must have 3 dimensions"
    assert YY.shape[-1] == 3, "YY must have 3 dimensions"
    if src_geom is None:
        raise ValueError("src_geom is required for the Laplace double-layer kernel")

    diff, dist2 = _pairwise_diff_dist2(XX, YY)
    if assume_not_self_interaction:
        dist = np.sqrt(dist2)
        rotn = np.sum(diff * src_geom[None, :, :], axis=-1)
        return -(1.0 / (4.0 * np.pi)) * rotn / (dist2 * dist)

    zero_mask = dist2 == 0
    if np.any(zero_mask):
        dist2[zero_mask] = 1.0
    dist = np.sqrt(dist2)
    rotn = np.sum(diff * src_geom[None, :, :], axis=-1)
    return -(1.0 / (4.0 * np.pi)) * rotn / (dist2 * dist)

def laplace_3d_adjoint_double_layer(XX, YY, param=0, trg_geom=None, assume_not_self_interaction=False):
    r"""
    Compute the 3D Laplace adjoint double-layer kernel
    D*(x,y) = \partial_{n_x} (1 / (4 pi |x-y|)).

    Parameters:
    - XX: (M, 3) target points
    - YY: (N, 3) source points
    - param: Not used, kept for API consistency
    - trg_geom: (M, 3) target normals/dipole directions

    Returns:
    - Dense interaction matrix of shape (M, N).
    """
    assert param == 0, "Parameter must be 0 for Laplace adjoint double-layer kernel"
    assert XX.shape[-1] == 3, "XX must have 3 dimensions"
    assert YY.shape[-1] == 3, "YY must have 3 dimensions"
    if trg_geom is None:
        raise ValueError("trg_geom is required for the Laplace adjoint double-layer kernel")

    diff, dist2 = _pairwise_diff_dist2(XX, YY)
    if assume_not_self_interaction:
        dist = np.sqrt(dist2)
        rotn = np.sum(diff * trg_geom[:, None, :], axis=-1)
        return (1.0 / (4.0 * np.pi)) * rotn / (dist2 * dist)

    zero_mask = dist2 == 0
    if np.any(zero_mask):
        dist2[zero_mask] = 1.0
    dist = np.sqrt(dist2)
    rotn = np.sum(diff * trg_geom[:, None, :], axis=-1)
    return (1.0 / (4.0 * np.pi)) * rotn / (dist2 * dist)


def helmholtz_3d_double_layer(XX, YY, param, src_geom=None, assume_not_self_interaction=False):
    r"""
    Compute the 3D Helmholtz double-layer kernel for the outgoing Green's
    function G_k(x,y) = exp(i k |x-y|) / (4 pi |x-y|), i.e.
    D_k(x,y) = \partial_{n_y} G_k(x,y).

    Parameters:
    - XX: (M, 3) target points
    - YY: (N, 3) source points
    - param: Helmholtz wavenumber
    - src_geom: (N, 3) source normals/dipole directions

    Returns:
    - Dense interaction matrix of shape (M, N).
    """
    assert param > 0, "Parameter must be positive for Helmholtz double-layer kernel"
    assert XX.shape[-1] == 3, "XX must have 3 dimensions"
    assert YY.shape[-1] == 3, "YY must have 3 dimensions"
    if src_geom is None:
        raise ValueError("src_geom is required for the Helmholtz double-layer kernel")

    diff, dist2 = _pairwise_diff_dist2(XX, YY)
    if assume_not_self_interaction:
        dist = np.sqrt(dist2)
        rotn = np.sum(diff * src_geom[None, :, :], axis=-1)
        kr = param * dist
        factor = np.exp(1j * kr) * (1j * kr - 1.0) / (dist2 * dist)
        return (1.0 / (4.0 * np.pi)) * rotn * factor

    zero_mask = dist2 == 0
    if np.any(zero_mask):
        dist2[zero_mask] = 1.0
    dist = np.sqrt(dist2)
    rotn = np.sum(diff * src_geom[None, :, :], axis=-1)

    kr = param * dist
    factor = np.exp(1j * kr) * (1j * kr - 1.0) / (dist2 * dist)
    return (1.0 / (4.0 * np.pi)) * rotn * factor


def helmholtz_3d_adjoint_double_layer(XX, YY, param, trg_geom=None, assume_not_self_interaction=False):
    r"""
    Compute the 3D Helmholtz adjoint double-layer kernel for the outgoing
    Green's function, i.e. D*_k(x,y) = \partial_{n_x} G_k(x,y).

    Parameters:
    - XX: (M, 3) target points
    - YY: (N, 3) source points
    - param: Helmholtz wavenumber
    - trg_geom: (M, 3) target normals/dipole directions

    Returns:
    - Dense interaction matrix of shape (M, N).
    """
    assert param > 0, "Parameter must be positive for Helmholtz adjoint double-layer kernel"
    assert XX.shape[-1] == 3, "XX must have 3 dimensions"
    assert YY.shape[-1] == 3, "YY must have 3 dimensions"
    if trg_geom is None:
        raise ValueError("trg_geom is required for the Helmholtz adjoint double-layer kernel")

    diff, dist2 = _pairwise_diff_dist2(XX, YY)
    if assume_not_self_interaction:
        dist = np.sqrt(dist2)
        rotn = np.sum(diff * trg_geom[:, None, :], axis=-1)
        kr = param * dist
        factor = np.exp(1j * kr) * (1j * kr - 1.0) / (dist2 * dist)
        return -(1.0 / (4.0 * np.pi)) * rotn * factor

    zero_mask = dist2 == 0
    if np.any(zero_mask):
        dist2[zero_mask] = 1.0
    dist = np.sqrt(dist2)
    rotn = np.sum(diff * trg_geom[:, None, :], axis=-1)

    kr = param * dist
    factor = np.exp(1j * kr) * (1j * kr - 1.0) / (dist2 * dist)
    return -(1.0 / (4.0 * np.pi)) * rotn * factor

######################################### SKEL UTILITY FUNCTIONS #########################################

@controller.wrap(limits=1, user_api='blas')
@controller.wrap(limits=1, user_api='openmp')
def skel_box_info(args):
    if len(args) == 10:
        item_idx, box, operator, kappa, XX_B, geom_B, box_center, box_len, nproxy, tol = args
    elif len(args) == 9:
        item_idx, box, operator, kappa, XX_B, box_center, box_len, nproxy, tol = args
        geom_B = None
    else:
        raise ValueError("Unexpected skel_box_info argument format")

    rank_box, idx_box, proj_box = skel_box_helper(
        operator,
        kappa,
        XX_B,
        box_center,
        box_len,
        tol,
        nproxy=nproxy,
        geom=geom_B,
    )
    return (item_idx, box, rank_box, idx_box, proj_box)

def skel_box_helper(operator, kappa, XX_box, box_center, box_length, rank_or_eps,
                    nproxy, geom=None):
    """
    Helper function for skeletonization of a box.
    
    Parameters:
    - operator: Kernel operator for interactions
    - kappa: Kernel parameter
    - XX_box: Points in the box
    - geom: General geometry object associated with the box points
    - box_center: Center of the box
    - box_length: Length of the box
    - rank_or_eps: Fixed rank or relative approximation tolerance
    - nproxy: Number of proxy points
    
    Returns:
    - rank: Computed rank
    - idx: Selected indices
    - proj: Projection matrix
    """
    from . import operators as operator_module

    operator = operator_module._require_operator(operator)
    K_stacked = build_stacked_proxy_matrix(
        operator,
        XX_box,
        box_center,
        box_length,
        nproxy,
        geom=geom,
        param=kappa,
    )
    rank, idx, proj = _skeleton_id_pivoted_qr(K_stacked, rank_or_eps)
    return rank, idx, proj
