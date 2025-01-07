from threadpoolctl import threadpool_info, ThreadpoolController
from simpletree import get_cube_surface
import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.linalg.interpolative import interp_decomp
from scipy.special import j0, y0
from scipy.linalg import lstsq
from time import time

# Initialize a controller for managing thread pool settings
controller = ThreadpoolController()

######################################### KERNEL FUNCTIONS #########################################

def l2dist(XX, YY):
    """
    Compute the Euclidean distance between two sets of points.
    
    Parameters:
    - XX: (N, d) array-like, first set of points
    - YY: (M, d) array-like, second set of points
    
    Returns:
    - Euclidean distance matrix of shape (N, M).
    """
    assert XX.shape[-1] <= 3, "XX must have dimension <= 3"
    assert YY.shape[-1] <= 3, "YY must have dimension <= 3"
    return scipy_cdist(XX, YY, metric='euclidean')

def laplace_2d(XX, YY, param=0):
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
    
    ZZ = l2dist(XX, YY)
    ZZ[ZZ == 0] = 1  # Avoid division by zero
    result = np.log(ZZ, out=ZZ)
    return result

def laplace_3d(XX, YY, param=0):
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
    
    ZZ = l2dist(XX, YY)
    tmp_bool = (ZZ == 0)
    ZZ[tmp_bool] = 1  # Avoid division by zero
    result = np.reciprocal(ZZ, out=ZZ)
    result[tmp_bool] = 0
    return result

def helmholtz_2d(XX, YY, param):
    """
    Compute the 2D Helmholtz kernel function.
    
    Parameters:
    - XX: (N, 2) array-like, first set of points
    - YY: (M, 2) array-like, second set of points
    - param: Helmholtz parameter
    
    Returns:
    - Helmholtz potential matrix of shape (N, M).
    """
    ZZ = l2dist(XX, YY)
    inds_zeros = np.where(ZZ == 0)
    ZZ[inds_zeros] += 1  # Avoid division by zero
    ZZ *= param

    result = y0(ZZ)
    result = result.astype(np.cdouble)
    result *= 1j  # Complex term
    result += j0(ZZ, out=ZZ)
    result[inds_zeros] = 0
    return result

def helmholtz_3d(XX, YY, param):
    """
    Compute the 3D Helmholtz kernel function.
    
    Parameters:
    - XX: (N, 3) array-like, first set of points
    - YY: (M, 3) array-like, second set of points
    - param: Helmholtz parameter
    
    Returns:
    - Helmholtz potential matrix of shape (N, M).
    """
    ZZ = l2dist(XX, YY)
    inds_zeros = np.where(ZZ == 0)

    ZZ_double = ZZ.astype(np.cdouble)
    numerator = np.exp(+1j * param * ZZ_double, out=ZZ_double)

    ZZ[inds_zeros] = float('inf')  # Avoid division by zero
    result = np.divide(numerator, ZZ, out=numerator)
    result[inds_zeros] = 0
    return result

######################################### SKEL UTILITY FUNCTIONS #########################################

@controller.wrap(limits=1, user_api='blas')
@controller.wrap(limits=1, user_api='openmp')
def skel_box(args):
    """
    Compute skeletonization for a single box.
    
    Parameters:
    - args: Tuple containing box parameters and kernel info
    
    Returns:
    - Tuple containing box results (box, coordinates, rank, indices, projection matrix).
    """
    (box, kernel_func, kappa, XX_B, bs, box_center, box_len, idx, nproxy, tol) = args
    rank_box, idx_box, proj_box = skel_box_helper(kernel_func, kappa, XX_B[:bs],
                                                  box_center, box_len, tol,
                                                  nproxy=nproxy, idx=idx)
    return (box, XX_B[:bs], rank_box, idx_box, proj_box)

def skel_box_helper(kernel_func, kappa, XX_box, box_center, box_length, rank_or_eps,
                    nproxy, idx=None, abs_tol=False):
    """
    Helper function for skeletonization of a box.
    
    Parameters:
    - kernel_func: Kernel function for interactions
    - kappa: Kernel parameter
    - XX_box: Points in the box
    - box_center: Center of the box
    - box_length: Length of the box
    - rank_or_eps: Rank or approximation tolerance
    - nproxy: Number of proxy points
    - idx: Indices for skeletonization (optional)
    - abs_tol: Whether to use absolute tolerance
    
    Returns:
    - rank: Computed rank
    - idx: Selected indices
    - proj: Projection matrix
    """
    XX_proxy = get_cube_surface(box_center, 2.95 * box_length, nproxy=nproxy)

    if kappa > 0:
        K_BP = kernel_func(XX_box, XX_proxy, kappa)
        K_PB = kernel_func(XX_proxy, XX_box, kappa)
    else:
        K_BP = kernel_func(XX_box, XX_proxy)
        K_PB = kernel_func(XX_proxy, XX_box)

    K_stacked = np.vstack((K_PB, K_BP.T))
    tmp_param = (kappa * box_length) * np.pi

    if idx is None:
        if rank_or_eps < 1:
            if abs_tol:
                norm = np.linalg.norm(K_stacked, ord=2)
                rank_or_eps = rank_or_eps / norm
                assert rank_or_eps < 1
            elif kappa > 0 and tmp_param > 1:
                rank_or_eps = rank_or_eps / tmp_param
            rank, idx, proj = interp_decomp(K_stacked, rank_or_eps, rand=True)
        else:
            rank = rank_or_eps
            idx, proj = interp_decomp(K_stacked, rank, rand=False)
    elif rank_or_eps >= 1 and idx is not None:
        rank = int(rank_or_eps)
        proj = lstsq(K_stacked[:, idx[:rank]], K_stacked[:, idx[rank:]],
                     lapack_driver='gelsy', overwrite_a=True, overwrite_b=True)[0]
    else:
        raise ValueError("Invalid arguments for skeletonization.")
    return rank, idx, proj

def verbose_check(fmm, tol, verbose=False):
    """
    Check the accuracy of skeletonization with optional verbosity.
    
    Parameters:
    - fmm: FMM object containing tree and kernel data
    - tol: Tolerance for accuracy
    - verbose: Whether to print detailed debug information (default: False)
    
    Returns:
    - None. Prints any errors exceeding tolerance.
    """
    for box in range(fmm.nboxes - 1, -1, -1):
        box_level = fmm.tree.get_box_level(box)
        if box_level < fmm.root_level:
            continue

        XX_B, bs, rank, idx, proj = fmm.get_skel_box_info(box)
        assert rank > 0

        I_N = fmm.tree.get_adjacent_inds(box)  # Indices of neighboring points
        I_F = np.setdiff1d(np.arange(fmm.N), I_N)  # Indices of far-field points

        K_BF = fmm.kernel_func(XX_B, fmm.tree.XX[I_F], fmm.kappa)
        K_FB = fmm.kernel_func(fmm.tree.XX[I_F], XX_B, fmm.kappa)

        err1 = np.linalg.norm(proj.T @ K_BF[:rank] - K_BF[rank:])
        err2 = np.linalg.norm(K_FB[:, :rank] @ proj - K_FB[:, rank:])

        err = np.min([err1, err2])
        if verbose:
            tmp = np.hstack((K_FB, K_BF.T))
            D = np.linalg.svd(tmp, compute_uv=False)
            print("rank true %d, rank computed %d" % (np.sum(D > D[0] * tol), rank))
        if err > 5 * tol:
            print("box %d on level %d has err %5.5e" % (box, box_level, err))
