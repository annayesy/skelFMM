import numpy as np
import pytest
import torch
from simpletree import BalancedTree, get_point_dist

from skelfmm import RecursiveSkeletonization, SkelFMM, SkelFMMBatched, operators, util, util_batched


def test_zero_rowcol_bnd():
    b = 5
    m = 50
    n = 40

    row_bnd = torch.tensor([5, 10, 5, 15, 20], dtype=int)

    tensor = torch.rand(b, m, n)
    tensor = util_batched.zero_row_bnd(tensor, row_bnd)

    for i in range(b):
        assert torch.linalg.norm(tensor[i, row_bnd[i]:]) == 0
        assert torch.linalg.norm(tensor[i, :row_bnd[i]]) > 0

    col_bnd = torch.tensor([10, 8, 5, 20, 10], dtype=int)

    tensor = torch.rand(b, m, n)
    tensor = util_batched.zero_col_bnd(tensor, col_bnd)

    for i in range(b):
        assert torch.linalg.norm(tensor[i, :, col_bnd[i]:]) == 0
        assert torch.linalg.norm(tensor[i, :, :col_bnd[i]]) > 0


def test_get_bytes_available_cpu_fallback():
    assert util_batched.get_bytes_available(torch.device("cpu")) > 0


def test_fast_apply_build_method_and_lazy_matvec_message(capsys):
    xx = get_point_dist(120, "square")
    apply = SkelFMM(xx, operators.LAPLACE_2D, tol=1e-5, leaf_size=20, defer_build=True)

    assert apply.build_time_seconds == 0.0
    assert apply.rank_max == 0

    random = np.random.RandomState(0)
    q = random.randn(xx.shape[0])
    y = apply.apply(q)

    captured = capsys.readouterr()
    assert "Building SkelFMM on first matvec" in captured.out
    assert y.shape == (xx.shape[0],)
    assert apply.build_time_seconds > 0.0
    assert apply.rank_max >= 0

    benchmark = apply.benchmark_matvec(nrepeat=1, warmup=0, seed=0)
    assert benchmark["min_seconds"] >= 0.0


def _clustered_points(ndim, npts):
    rng = np.random.default_rng(1234 + ndim + npts)
    base = rng.random((npts, ndim)) * 0.15
    offsets = np.linspace(1e-6, npts * 1e-6, npts, dtype=np.float64)[:, None]
    direction = np.arange(1, ndim + 1, dtype=np.float64)[None, :]
    return base + offsets * direction


def _build_serial_and_batched_fmm(
    points,
    operator,
    *,
    kappa=0.0,
    tol=1e-6,
    leaf_size=40,
    geometry=None,
    use_single_precision=False,
    npoints_max=None,
    p=None,
):
    tree = BalancedTree(points, leaf_size=leaf_size)
    fmm = RecursiveSkeletonization(tree, operator, kappa=kappa, tol=tol, geometry=geometry)
    if npoints_max is None:
        npoints_max = points.shape[0]
    fmm.skel_tree(npoints_max=npoints_max, p=p, verbose=False)
    fmm.setup_lists()

    batched_fmm = SkelFMMBatched(tree, operator, kappa=kappa)
    batched_fmm.build_from_copy(fmm, use_single_precision=use_single_precision)
    return fmm, batched_fmm


def test_exact_distance_helpers_and_masks():
    xx2 = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
    yy2 = torch.tensor([[[0.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    sq = util_batched.square_dist_helper_exact(xx2, yy2)
    assert torch.allclose(sq, torch.tensor([[[0.0, 4.0], [1.0, 5.0]]], dtype=torch.float64))

    zz2 = torch.ones(2, 2, dtype=torch.float64)
    masked2 = util_batched.set_self_interactions_to_value(zz2.clone(), True, value=7.0)
    assert torch.allclose(torch.diag(masked2), torch.full((2,), 7.0, dtype=torch.float64))

    zz3 = torch.ones(1, 2, 2, dtype=torch.float64)
    masked3 = util_batched.set_self_interactions_to_value(zz3.clone(), True, value=5.0)
    assert torch.allclose(torch.diagonal(masked3[0]), torch.full((2,), 5.0, dtype=torch.float64))
    assert torch.allclose(util_batched.set_self_interactions_to_value(zz3.clone(), False), zz3)

    xx32 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    dist32 = util_batched.cdist(xx32, xx32, True)
    assert torch.all(dist32 > 0)


def test_batched_build_from_copy_preserves_metadata_and_active_entries():
    xx = get_point_dist(150, "square")
    fmm, batched_fmm = _build_serial_and_batched_fmm(xx, operators.LAPLACE_2D, tol=1e-6, leaf_size=40)

    assert batched_fmm.N == fmm.N
    assert batched_fmm.nlevels == fmm.nlevels
    assert batched_fmm.nboxes == fmm.nboxes
    assert batched_fmm.max_bs == fmm.max_bs
    assert batched_fmm.max_rank == fmm.max_rank
    assert batched_fmm.min_rank == fmm.min_rank
    assert batched_fmm.root_level == fmm.root_level
    assert batched_fmm.tol == fmm.tol

    assert np.array_equal(batched_fmm.bs_list.cpu().numpy(), fmm.bs_list)
    assert np.array_equal(batched_fmm.rank_list.cpu().numpy(), fmm.rank_list)
    assert np.array_equal(batched_fmm.tree_boxes.cpu().numpy(), fmm.tree_boxes)
    assert np.array_equal(batched_fmm.tree_lev_sep.cpu().numpy(), fmm.tree_lev_sep)

    xx_list_batched = batched_fmm.XX_list.cpu().numpy()
    xx_rank_batched = batched_fmm.XX_rank_list.cpu().numpy()
    for box in range(fmm.nboxes):
        bs = int(fmm.bs_list[box])
        rank = int(fmm.rank_list[box])
        assert np.allclose(xx_list_batched[box, :bs], fmm.XX_list[box, :bs])
        assert np.allclose(xx_rank_batched[box, :rank], fmm.XX_list[box, :rank])

    assert batched_fmm.nbytes_proj() >= 0
    assert batched_fmm.nbytes_lists() > 0


def _kernel_reference_check(
    reference_kernel,
    batched_kernel,
    xx,
    yy,
    q_np,
    kappa,
    self_dist_bool,
    atol,
    rtol,
):
    ref_mv = reference_kernel(xx, yy, kappa) @ q_np
    xx_t = torch.tensor(xx, dtype=torch.float64).unsqueeze(0)
    yy_t = torch.tensor(yy, dtype=torch.float64).unsqueeze(0)
    q_dtype = torch.complex128 if np.iscomplexobj(q_np) else torch.float64
    q_t = torch.tensor(q_np, dtype=q_dtype).unsqueeze(0)

    result = batched_kernel(q_t, xx_t, yy_t, kappa, self_dist_bool)[0].numpy()
    assert np.allclose(result, ref_mv, atol=atol, rtol=rtol)


def _kernel_keops_check(lazy_kernel, batched_kernel, xx, yy, q_np, kappa, self_dist_bool, atol, rtol):
    xx_t = torch.tensor(xx, dtype=torch.float64).unsqueeze(0)
    yy_t = torch.tensor(yy, dtype=torch.float64).unsqueeze(0)
    q_dtype = torch.complex128 if np.iscomplexobj(q_np) else torch.float64
    q_t = torch.tensor(q_np, dtype=q_dtype).unsqueeze(0)

    lazy_out = lazy_kernel(q_t, xx_t, yy_t, self_dist_bool) if kappa == 0 else lazy_kernel(q_t, xx_t, yy_t, kappa, self_dist_bool)
    exact_out = batched_kernel(q_t, xx_t, yy_t, kappa, self_dist_bool)
    assert torch.allclose(exact_out, lazy_out, atol=atol, rtol=rtol)


def _double_layer_keops_check(lazy_kernel, batched_kernel, xx, yy, geom_trg, geom_src, q_np, self_dist_bool, atol, rtol):
    xx_t = torch.tensor(xx, dtype=torch.float64).unsqueeze(0)
    yy_t = torch.tensor(yy, dtype=torch.float64).unsqueeze(0)
    q_t = torch.tensor(q_np, dtype=torch.float64).unsqueeze(0)
    geom_trg_t = None if geom_trg is None else torch.tensor(geom_trg, dtype=torch.float64).unsqueeze(0)
    geom_src_t = None if geom_src is None else torch.tensor(geom_src, dtype=torch.float64).unsqueeze(0)

    lazy_out = lazy_kernel(
        q_t,
        xx_t,
        yy_t,
        self_dist_bool,
        trg_geom=geom_trg_t,
        src_geom=geom_src_t,
    )
    exact_out = batched_kernel(
        q_t,
        xx_t,
        yy_t,
        0.0,
        self_dist_bool,
        trg_geom=geom_trg_t,
        src_geom=geom_src_t,
    )
    diff = torch.max(torch.abs(exact_out - lazy_out)).item()
    scale = max(torch.max(torch.abs(exact_out)).item(), 1e-30)
    assert diff / scale < max(atol, rtol)


@pytest.mark.parametrize(
    ("reference_kernel", "batched_kernel", "ndim", "kappa", "is_complex", "atol", "rtol"),
    [
        (util.laplace_2d, util_batched.laplace_2d, 2, 0.0, False, 1e-10, 1e-10),
        (util.laplace_3d, util_batched.laplace_3d, 3, 0.0, False, 1e-10, 1e-10),
        (util.helmholtz_2d, util_batched.helmholtz_2d, 2, 3.0, True, 1e-10, 1e-10),
        (util.helmholtz_3d, util_batched.helmholtz_3d, 3, 2.0, True, 1e-10, 1e-10),
    ],
)
def test_batched_kernels_match_reference_self_and_near_neighbor(
    reference_kernel,
    batched_kernel,
    ndim,
    kappa,
    is_complex,
    atol,
    rtol,
):
    xx = _clustered_points(ndim, 6)
    yy = xx + 5e-7

    q_rng = np.random.default_rng(2024 + ndim)
    q_np = q_rng.random((yy.shape[0], 1))
    if is_complex:
        q_np = q_np + 1j * q_rng.random((yy.shape[0], 1))

    _kernel_reference_check(
        reference_kernel,
        batched_kernel,
        xx,
        yy,
        q_np,
        kappa,
        self_dist_bool=False,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("lazy_kernel", "batched_kernel", "ndim", "kappa", "is_complex", "atol", "rtol"),
    [
        (util_batched.laplace_2d_lazy, util_batched.laplace_2d, 2, 0.0, False, 1e-10, 1e-10),
        (util_batched.laplace_3d_lazy, util_batched.laplace_3d, 3, 0.0, False, 1e-10, 1e-10),
        (util_batched.helmholtz_3d_lazy, util_batched.helmholtz_3d, 3, 2.0, True, 1e-10, 1e-10),
    ],
)
def test_batched_kernels_match_pykeops_lazy_self_and_near_neighbor(
    lazy_kernel,
    batched_kernel,
    ndim,
    kappa,
    is_complex,
    atol,
    rtol,
):
    pytest.importorskip("pykeops")

    xx = _clustered_points(ndim, 6)
    yy = xx + 5e-7

    q_rng = np.random.default_rng(4040 + ndim)
    q_np = q_rng.random((yy.shape[0], 1))
    if is_complex:
        q_np = q_np + 1j * q_rng.random((yy.shape[0], 1))

    _kernel_keops_check(lazy_kernel, batched_kernel, xx, yy, q_np, kappa, False, atol, rtol)
    _kernel_keops_check(lazy_kernel, batched_kernel, xx, xx, q_np, kappa, True, atol, rtol)


@pytest.mark.parametrize(
    ("lazy_kernel", "batched_kernel", "use_src_geom"),
    [
        (
            lambda q, xx, yy, self_bool, trg_geom=None, src_geom=None: util_batched.laplace_3d_double_layer_lazy(
                q, xx, yy, self_bool, src_geom=src_geom
            ),
            lambda q, xx, yy, _param, self_bool, trg_geom=None, src_geom=None: util_batched.laplace_3d_double_layer(
                q, xx, yy, 0.0, self_bool, src_geom=src_geom
            ),
            True,
        ),
        (
            lambda q, xx, yy, self_bool, trg_geom=None, src_geom=None: util_batched.laplace_3d_adjoint_double_layer_lazy(
                q, xx, yy, self_bool, trg_geom=trg_geom
            ),
            lambda q, xx, yy, _param, self_bool, trg_geom=None, src_geom=None: util_batched.laplace_3d_adjoint_double_layer(
                q, xx, yy, 0.0, self_bool, trg_geom=trg_geom
            ),
            False,
        ),
    ],
)
def test_double_layer_batched_kernels_match_pykeops_lazy_self_and_near_neighbor(
    lazy_kernel,
    batched_kernel,
    use_src_geom,
):
    pytest.importorskip("pykeops")

    xx = _clustered_points(3, 6)
    yy = xx + 5e-7
    normals = yy if use_src_geom else xx
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    q_np = np.random.default_rng(8080).random((yy.shape[0], 1))

    geom_trg = None if use_src_geom else normals
    geom_src = normals if use_src_geom else None

    _double_layer_keops_check(
        lazy_kernel,
        batched_kernel,
        xx,
        yy,
        geom_trg,
        geom_src,
        q_np,
        False,
        5e-5,
        5e-5,
    )
    _double_layer_keops_check(
        lazy_kernel,
        batched_kernel,
        xx,
        xx,
        geom_trg if geom_trg is None else xx / np.linalg.norm(xx, axis=1, keepdims=True),
        geom_src if geom_src is None else xx / np.linalg.norm(xx, axis=1, keepdims=True),
        q_np,
        True,
        5e-5,
        5e-5,
    )


@pytest.mark.parametrize(
    ("lazy_kernel", "batched_kernel", "use_src_geom"),
    [
        (
            lambda q, xx, yy, kappa, self_bool, trg_geom=None, src_geom=None: util_batched.helmholtz_3d_double_layer_lazy(
                q, xx, yy, kappa, self_bool, src_geom=src_geom
            ),
            lambda q, xx, yy, kappa, self_bool, trg_geom=None, src_geom=None: util_batched.helmholtz_3d_double_layer(
                q, xx, yy, kappa, self_bool, src_geom=src_geom
            ),
            True,
        ),
        (
            lambda q, xx, yy, kappa, self_bool, trg_geom=None, src_geom=None: util_batched.helmholtz_3d_adjoint_double_layer_lazy(
                q, xx, yy, kappa, self_bool, trg_geom=trg_geom
            ),
            lambda q, xx, yy, kappa, self_bool, trg_geom=None, src_geom=None: util_batched.helmholtz_3d_adjoint_double_layer(
                q, xx, yy, kappa, self_bool, trg_geom=trg_geom
            ),
            False,
        ),
    ],
)
def test_helmholtz_double_layer_batched_kernels_match_pykeops_lazy_self_and_near_neighbor(
    lazy_kernel,
    batched_kernel,
    use_src_geom,
):
    pytest.importorskip("pykeops")

    xx = _clustered_points(3, 6)
    yy = xx + 5e-7
    q_rng = np.random.default_rng(9090)
    q_np = q_rng.random((yy.shape[0], 1)) + 1j * q_rng.random((yy.shape[0], 1))

    near_normals = (yy if use_src_geom else xx)
    near_normals = near_normals / np.linalg.norm(near_normals, axis=1, keepdims=True)
    self_normals = xx / np.linalg.norm(xx, axis=1, keepdims=True)

    geom_trg = None if use_src_geom else near_normals
    geom_src = near_normals if use_src_geom else None
    self_geom_trg = None if use_src_geom else self_normals
    self_geom_src = self_normals if use_src_geom else None

    xx_t = torch.tensor(xx, dtype=torch.float64).unsqueeze(0)
    yy_t = torch.tensor(yy, dtype=torch.float64).unsqueeze(0)
    q_t = torch.tensor(q_np, dtype=torch.complex128).unsqueeze(0)
    geom_trg_t = None if geom_trg is None else torch.tensor(geom_trg, dtype=torch.float64).unsqueeze(0)
    geom_src_t = None if geom_src is None else torch.tensor(geom_src, dtype=torch.float64).unsqueeze(0)
    self_geom_trg_t = None if self_geom_trg is None else torch.tensor(self_geom_trg, dtype=torch.float64).unsqueeze(0)
    self_geom_src_t = None if self_geom_src is None else torch.tensor(self_geom_src, dtype=torch.float64).unsqueeze(0)

    lazy_out = lazy_kernel(q_t, xx_t, yy_t, 2.0, False, trg_geom=geom_trg_t, src_geom=geom_src_t)
    exact_out = batched_kernel(q_t, xx_t, yy_t, 2.0, False, trg_geom=geom_trg_t, src_geom=geom_src_t)
    rel = torch.max(torch.abs(exact_out - lazy_out)).item() / max(
        torch.max(torch.abs(exact_out)).item(), 1e-30
    )
    assert rel < 5e-5

    lazy_self = lazy_kernel(q_t, xx_t, xx_t, 2.0, True, trg_geom=self_geom_trg_t, src_geom=self_geom_src_t)
    exact_self = batched_kernel(q_t, xx_t, xx_t, 2.0, True, trg_geom=self_geom_trg_t, src_geom=self_geom_src_t)
    rel = torch.max(torch.abs(exact_self - lazy_self)).item() / max(
        torch.max(torch.abs(exact_self)).item(), 1e-30
    )
    assert rel < 5e-5


def test_batched_fmm_matches_reference_cpu():
    np.random.seed(0)
    xx = get_point_dist(150, "square")
    tree = BalancedTree(xx, leaf_size=40)
    fmm = RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=1e-6)
    fmm.skel_tree(npoints_max=150, p=None, verbose=False)
    fmm.setup_lists()

    batched_fmm = SkelFMMBatched(
        tree,
        operators.LAPLACE_2D,
        kappa=0,
    )
    batched_fmm.build_from_copy(fmm)

    x_np = np.random.default_rng(0).random(fmm.N)
    ref = fmm.matvec(x_np.copy())
    got = batched_fmm.matvec(torch.from_numpy(x_np).to(torch.float64)).cpu().numpy()
    assert np.allclose(got, ref, atol=1e-8, rtol=1e-8)


def test_batched_fmm_matches_reference_for_laplace_double_layer():
    xx = get_point_dist(120, "sphere_surface")
    normals = xx / np.linalg.norm(xx, axis=1, keepdims=True)
    fmm, batched_fmm = _build_serial_and_batched_fmm(
        xx,
        operators.LAPLACE_DOUBLE_LAYER_3D,
        tol=1e-5,
        leaf_size=40,
        geometry=normals,
    )

    x_np = np.random.default_rng(5).random(fmm.N)
    ref = fmm.matvec(x_np.copy())
    got = batched_fmm.matvec(torch.from_numpy(x_np).to(torch.float64)).cpu().numpy()
    rel = np.linalg.norm(got - ref, ord=np.inf) / max(np.linalg.norm(ref, ord=np.inf), 1e-30)
    assert rel < 1e-5


def test_batched_annulus_compression_is_nontrivial_and_matches_serial():
    xx = get_point_dist(4000, "curvy_annulus")
    fmm, batched_fmm = _build_serial_and_batched_fmm(
        xx,
        operators.LAPLACE_2D,
        tol=1e-5,
        leaf_size=100,
        npoints_max=500,
        p="auto",
    )

    bs = batched_fmm.bs_list.cpu().numpy()
    rk = batched_fmm.rank_list.cpu().numpy()
    active = (bs > 0) & (rk > 0)
    strictly_compressed = active & (rk < bs)

    assert np.count_nonzero(active) > 0
    assert np.count_nonzero(strictly_compressed) >= int(0.8 * np.count_nonzero(active))
    assert np.max(rk) < 0.5 * np.max(bs)
    assert np.mean(rk[active] / bs[active]) < 0.6

    x_np = np.random.default_rng(17).random(fmm.N)
    ref = fmm.matvec(x_np.copy())
    got = batched_fmm.matvec(torch.from_numpy(x_np).to(torch.float64)).cpu().numpy()
    assert np.allclose(got, ref, atol=1e-8, rtol=1e-8)


def test_batched_fmm_single_precision_helmholtz():
    np.random.seed(1)
    xx = get_point_dist(120, "cube")
    tree = BalancedTree(xx, leaf_size=40)
    fmm = RecursiveSkeletonization(tree, operators.HELMHOLTZ_3D, kappa=2.0, tol=1e-5)
    fmm.skel_tree(npoints_max=120, p=None, verbose=False)
    fmm.setup_lists()

    batched_fmm = SkelFMMBatched(
        tree,
        operators.HELMHOLTZ_3D,
        kappa=2.0,
    )
    batched_fmm.build_from_copy(fmm, use_single_precision=True)

    assert batched_fmm.XX_list.dtype == torch.float32
    assert batched_fmm.proj_list.dtype == torch.complex64
    assert batched_fmm.proj_list_tree.dtype == torch.complex64

    x = torch.from_numpy(np.random.default_rng(0).random(fmm.N)).to(torch.float32)
    result = batched_fmm.matvec(x)
    assert result.dtype == torch.complex64

    ref = fmm.matvec(x.numpy().copy())
    got = result.cpu().numpy().astype(np.complex128)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 5e-4


def test_batched_nbytes_helpers_and_grouped_translation_paths():
    np.random.seed(2)
    xx = get_point_dist(150, "square")
    tree = BalancedTree(xx, leaf_size=40)
    fmm = RecursiveSkeletonization(tree, operators.LAPLACE_2D, kappa=0, tol=1e-6)
    fmm.skel_tree(npoints_max=150, p=None, verbose=False)
    fmm.setup_lists()

    batched_fmm = SkelFMMBatched(
        tree,
        operators.LAPLACE_2D,
        kappa=0,
    )
    batched_fmm.build_from_copy(fmm)

    assert batched_fmm.nbytes_proj() >= 0
    assert batched_fmm.nbytes_lists() > 0

    q_vec = torch.randn(batched_fmm.nboxes, batched_fmm.max_bs, dtype=torch.float64)

    uq_pairs = batched_fmm.leaf_uskel_from_qskel_list[:2]
    if uq_pairs.shape[0] > 0:
        u_vec_plain = torch.zeros_like(q_vec)
        u_vec_grouped = torch.zeros_like(q_vec)
        batched_fmm.translation(uq_pairs, u_vec_plain, None, q_vec, None, batched_fmm.max_bs_leaf, verbose=False)
        batched_fmm.translation(uq_pairs.unsqueeze(1), u_vec_grouped, None, q_vec, None, batched_fmm.max_bs_leaf, verbose=False)
        assert torch.allclose(u_vec_plain, u_vec_grouped)

        u_vec_plain = torch.zeros_like(q_vec)
        u_vec_grouped = torch.zeros_like(q_vec)
        u_skel_plain = torch.zeros_like(q_vec)
        u_skel_grouped = torch.zeros_like(q_vec)
        q_skel = torch.randn_like(q_vec)
        batched_fmm.translation(uq_pairs, u_vec_plain, u_skel_plain, q_vec, q_skel, batched_fmm.max_rank, verbose=False)
        batched_fmm.translation(uq_pairs.unsqueeze(1), u_vec_grouped, u_skel_grouped, q_vec, q_skel, batched_fmm.max_rank, verbose=False)
        assert torch.allclose(u_vec_plain, u_vec_grouped)
        assert torch.allclose(u_skel_plain, u_skel_grouped)
