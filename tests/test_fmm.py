from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from simpletree import BalancedTree, get_point_dist, morton

from skelfmm import operators, recursive_skel, util, util_batched


def add_patches(
    ax,
    tree,
    box_list,
    keys=False,
    edgecolor="black",
    facecolor="white",
    alpha=0.7,
    linewidth=2.0,
    text_label=False,
    fontsize=20,
):
    for box in box_list:
        if keys:
            c, length = morton.get_key_params(box, tree.c0, tree.L0)
        else:
            c = tree.get_box_center(box)
            length = tree.get_box_length(box)
        rect = patches.Rectangle(
            (c[0] - length / 2, c[1] - length / 2),
            length,
            length,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
        )
        if (not keys) and text_label:
            ax.text(
                c[0],
                c[1],
                "${%d}$" % box,
                horizontalalignment="center",
                fontsize=fontsize,
            )
        ax.add_patch(rect)


def get_problem_setup(n, npoints_max, output_dir: Path, tol=1e-5):
    np.random.seed(0)
    xx1 = get_point_dist(n, "square")
    xx1 = xx1 * 0.49

    xx2 = get_point_dist(80, "square")
    xx2 = xx2 * 0.49
    xx2 += 0.51

    xx = np.vstack((xx1, xx2))

    tree = BalancedTree(xx, leaf_size=100)
    fmm = recursive_skel.RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=tol)

    fmm.skel_tree(npoints_max=npoints_max, verbose=True)
    fmm.setup_lists()

    fig, ax = plt.subplots()
    ax.scatter(xx[:, 0], xx[:, 1], s=4, c="tab:blue")
    ax.axis("off")

    add_patches(
        ax,
        tree,
        tree.get_leaves(),
        keys=False,
        edgecolor="tab:gray",
        facecolor="none",
        text_label=True,
        linewidth=2.0,
        alpha=0.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(output_dir / "test_case.png", bbox_inches="tight")
    plt.close(fig)
    return fmm


def get_problem_setup_annulus(n, npoints_max, tol=1e-5):
    np.random.seed(0)
    xx = get_point_dist(n, "curvy_annulus")

    tree = BalancedTree(xx, leaf_size=100)
    fmm = recursive_skel.RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=tol)

    fmm.skel_tree(npoints_max=npoints_max, verbose=True)
    fmm.setup_lists()
    return fmm


def err_eval(indices, fmm):
    x = np.random.rand(fmm.N)
    relerr_inf = fmm.relerr_fmm_apply_check(x, indices=indices)
    assert relerr_inf < fmm.tol


def skel_setup(npoints_max, tmp_path: Path):
    fmm = get_problem_setup(400, npoints_max, tmp_path)
    root_level = 2
    assert fmm.root_level == root_level

    nleaves_above = 0
    for box in fmm.root_boxes:
        if fmm.tree.get_box_level(box) < root_level:
            nleaves_above += 1
    assert nleaves_above == 1

    skel_boxes = np.setdiff1d(np.arange(fmm.nboxes), fmm.root_boxes)

    for box in fmm.root_boxes:
        assert fmm.rank_list[box] == 0
        assert np.linalg.norm(fmm.proj_list[box]) == 0

    for box in skel_boxes:
        if fmm.tree.get_box_level(box) <= root_level:
            continue
        assert fmm.rank_list[box] > 0

    for left, right in fmm.u_from_qskel_list:
        assert fmm.tree.get_box_level(left) >= root_level
        assert fmm.tree.get_box_level(right) > root_level

    for top, leaf in fmm.uskel_from_q_list:
        assert fmm.tree.get_box_level(leaf) >= root_level
        assert fmm.tree.get_box_level(top) > root_level

    for top1, top2 in fmm.uskel_from_qskel_list:
        assert fmm.tree.get_box_level(top1) > root_level
        assert fmm.tree.get_box_level(top2) > root_level
    return fmm


def test_leafabove(tmp_path):
    fmm = skel_setup(400, tmp_path)
    leaf_above = fmm.root_boxes[0]
    indices = fmm.tree.get_box_inds(leaf_above)
    err_eval(indices, fmm)


def test_rand(tmp_path):
    fmm = skel_setup(400, tmp_path)
    indices = np.random.choice(fmm.N, 10, replace=False)
    err_eval(indices, fmm)


def test_annulus():
    fmm = get_problem_setup_annulus(3000, 100)
    indices = np.random.choice(fmm.N, 10, replace=False)
    err_eval(indices, fmm)


def test_kernel_eval_is_exact_and_fmm_carries_approximation(tmp_path):
    fmm = get_problem_setup(400, 100, tmp_path, tol=1e-5)

    xx = fmm.tree.XX
    q = np.random.default_rng(0).random((fmm.N, 1))
    truevec = util.laplace_2d(xx, xx) @ q

    xx_t = torch.from_numpy(xx).to(torch.float64).unsqueeze(0)
    q_t = torch.from_numpy(q).to(torch.float64).unsqueeze(0)
    kernel_vec = util_batched.laplace_2d(q_t, xx_t, xx_t, 0.0, True)[0].numpy()
    fmm_vec = fmm.matvec(q[:, 0].copy()).reshape(-1, 1)

    kernel_rel = np.linalg.norm(kernel_vec - truevec) / np.linalg.norm(truevec)
    fmm_rel = np.linalg.norm(fmm_vec - truevec) / np.linalg.norm(truevec)

    assert kernel_rel < 1e-12
    assert fmm_rel < 10 * fmm.tol
    assert kernel_rel < 1e-2 * fmm_rel


def test_constructor_explicit_limits_and_nbytes_proj():
    xx = get_point_dist(80, "square")
    tree = BalancedTree(xx, leaf_size=10)
    fmm = recursive_skel.RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=1e-6, max_bs=12, max_rank=7)

    assert fmm.max_bs == 12
    assert fmm.max_rank == 7
    assert fmm.max_rank_leaf == 7
    assert fmm.nbytes_proj > 0


def test_set_skel_box_info_grows_leaf_and_tree_projection_storage():
    xx = get_point_dist(200, "square")
    tree = BalancedTree(xx, leaf_size=20)
    fmm = recursive_skel.RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=1e-6, max_bs=20, max_rank=1)

    leaves = tree.get_leaves()
    leaf = next(box for box in leaves if fmm.bs_list[box] >= 2)
    bs_leaf = int(fmm.bs_list[leaf])
    rank_leaf = 2
    xx_leaf = fmm.XX_list[leaf, :bs_leaf].copy()
    idx_leaf = np.arange(bs_leaf)
    proj_leaf = np.zeros((rank_leaf, bs_leaf - rank_leaf), dtype=fmm.proj_list.dtype)

    old_leaf_rank = fmm.max_rank_leaf
    fmm.set_skel_box_info(leaf, xx_leaf, rank_leaf, idx_leaf, proj_leaf)
    assert fmm.max_rank_leaf > old_leaf_rank

    assert fmm.tree_boxes.shape[0] > 1
    box = fmm.tree_boxes[1]
    xx_box = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.25]])
    rank_box = 2
    idx_box = np.array([0, 1, 2])
    proj_box = np.zeros((rank_box, xx_box.shape[0] - rank_box), dtype=fmm.proj_list_tree.dtype)

    old_tree_rank = fmm.max_rank
    fmm.set_skel_box_info(box, xx_box, rank_box, idx_box, proj_box)
    assert fmm.max_rank > old_tree_rank


def test_update_lev_skel_grows_max_bs():
    xx = get_point_dist(80, "square")
    tree = BalancedTree(xx, leaf_size=2)
    fmm = recursive_skel.RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=1e-6, max_bs=2, max_rank=2)

    deepest_internal_level = tree.nlevels - 2
    internal_boxes = [box for box in tree.get_boxes_level(deepest_internal_level) if not tree.is_leaf(box)]
    assert internal_boxes

    for leaf in tree.get_leaves():
        fmm.rank_list[leaf] = fmm.bs_list[leaf]

    old_max_bs = fmm.max_bs
    fmm.update_lev_skel(deepest_internal_level)

    assert fmm.max_bs > old_max_bs
    assert any(fmm.bs_list[box] > old_max_bs for box in internal_boxes)


def test_skel_tree_auto_parallel():
    xx = get_point_dist(600, "square")
    tree = BalancedTree(xx, leaf_size=20)
    fmm = recursive_skel.RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=1e-5)

    fmm.skel_tree(npoints_max=100, p="auto", verbose=False, min_parallel_items=1)

    assert fmm.root_level >= 2
    assert np.any(fmm.rank_list > 0)
