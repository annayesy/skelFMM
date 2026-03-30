import numpy as np
import pytest

from skelfmm import operators, util


def test_kernel_diagonals_are_zero():
    xx2 = np.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.1]])
    xx3 = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.1], [0.75, 0.1, 0.2]])

    lap2 = util.laplace_2d(xx2, xx2)
    lap3 = util.laplace_3d(xx3, xx3)
    hel2 = util.helmholtz_2d(xx2, xx2, 2.0)
    hel3 = util.helmholtz_3d(xx3, xx3, 2.0)

    assert np.allclose(np.diag(lap2), 0.0)
    assert np.allclose(np.diag(lap3), 0.0)
    assert np.allclose(np.diag(hel2), 0.0)
    assert np.allclose(np.diag(hel3), 0.0)


def test_far_kernel_fast_paths_match_default():
    xx = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.3]])
    yy = np.array([[2.0, 0.5, 0.1], [3.0, -0.1, 0.7]])
    src_geom = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    trg_geom = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    assert np.allclose(
        util.laplace_3d(xx, yy, assume_not_self_interaction=True),
        util.laplace_3d(xx, yy),
    )
    assert np.allclose(
        util.helmholtz_3d(xx, yy, 1.7, assume_not_self_interaction=True),
        util.helmholtz_3d(xx, yy, 1.7),
    )
    assert np.allclose(
        util.laplace_3d_double_layer(xx, yy, src_geom=src_geom, assume_not_self_interaction=True),
        util.laplace_3d_double_layer(xx, yy, src_geom=src_geom),
    )
    assert np.allclose(
        util.helmholtz_3d_double_layer(xx, yy, 1.7, src_geom=src_geom, assume_not_self_interaction=True),
        util.helmholtz_3d_double_layer(xx, yy, 1.7, src_geom=src_geom),
    )
    assert np.allclose(
        util.laplace_3d_adjoint_double_layer(xx, yy, trg_geom=trg_geom, assume_not_self_interaction=True),
        util.laplace_3d_adjoint_double_layer(xx, yy, trg_geom=trg_geom),
    )
    assert np.allclose(
        util.helmholtz_3d_adjoint_double_layer(
            xx,
            yy,
            1.7,
            trg_geom=trg_geom,
            assume_not_self_interaction=True,
        ),
        util.helmholtz_3d_adjoint_double_layer(xx, yy, 1.7, trg_geom=trg_geom),
    )


def test_skel_box_helper_eps_and_fixed_rank_paths():
    rng = np.random.default_rng(0)
    xx_box = rng.random((8, 2)) * 0.4 - 0.2
    center = np.array([0.0, 0.0])

    rank, idx, proj = util.skel_box_helper(
        operators.LAPLACE_2D,
        0.0,
        xx_box,
        center,
        0.5,
        1e-6,
        nproxy=24,
    )

    assert 0 < rank <= xx_box.shape[0]
    assert idx.shape == (xx_box.shape[0],)
    assert proj.shape[0] == rank
    assert proj.shape[1] == xx_box.shape[0] - rank

    rank_fixed, idx_fixed, proj_fixed = util.skel_box_helper(
        operators.LAPLACE_2D,
        0.0,
        xx_box,
        center,
        0.5,
        rank,
        nproxy=24,
    )

    assert rank_fixed == rank
    assert np.array_equal(idx_fixed, idx)
    assert proj_fixed.shape == proj.shape


def test_skel_box_helper_helmholtz_path():
    rng = np.random.default_rng(4)
    xx_box = rng.random((6, 2)) * 0.5
    center = np.array([0.25, 0.25])

    rank, idx, proj = util.skel_box_helper(
        operators.HELMHOLTZ_2D,
        8.0,
        xx_box,
        center,
        1.0,
        1e-5,
        nproxy=24,
    )

    assert 0 < rank <= xx_box.shape[0]
    assert idx.shape == (xx_box.shape[0],)
    assert np.iscomplexobj(proj)


def test_pivoted_qr_id_reconstructs_column_skeletonization():
    random = np.random.RandomState(7)
    A = random.randn(18, 8)

    rank, idx, proj = util._skeleton_id_pivoted_qr(A.copy(), 1e-10)

    assert 0 < rank <= A.shape[1]
    assert idx.shape == (A.shape[1],)
    assert proj.shape == (rank, A.shape[1] - rank)

    skel = idx[:rank]
    red = idx[rank:]
    if red.size:
        approx = A[:, skel] @ proj
        assert np.linalg.norm(approx - A[:, red]) / np.linalg.norm(A[:, red]) < 1e-8
