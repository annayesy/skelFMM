from skelfmm.examples.convergence_laplace_double_layer_sphere import (
    estimate_convergence as estimate_laplace_double_layer_convergence,
    fast_self_apply_error as laplace_double_layer_self_apply_error,
)
from skelfmm.examples.convergence_laplace_single_layer_sphere import (
    estimate_convergence as estimate_laplace_convergence,
    fast_self_apply_error as laplace_single_layer_self_apply_error,
)
from skelfmm.examples.driver_single_layer_kernels import main as single_layer_demo_main


def test_laplace_single_layer_example_is_numerically_accurate():
    relerr = laplace_single_layer_self_apply_error(
        n_points=64,
        tol=1e-4,
        leaf_size=32,
        use_single_precision=True,
    )
    assert relerr < 1e-5

    report = estimate_laplace_convergence(
        n_points_list=(48, 96, 192),
        tol=1e-4,
        leaf_size=32,
        matvec_repeats=1,
        use_single_precision=True,
    )
    assert len(report["orders"]) == 1
    assert report["orders"][0] > 1.9
    assert max(report["rel_subset_error"]) < 1e-5
    assert all(err > 0 for err in report["errors"])


def test_laplace_double_layer_example_is_numerically_accurate():
    relerr = laplace_double_layer_self_apply_error(
        n_points=96,
        tol=1e-4,
        leaf_size=32,
        use_single_precision=True,
    )
    assert relerr < 1e-5

    report = estimate_laplace_double_layer_convergence(
        n_panels_list=(80, 320, 1280),
        tol=1e-4,
        leaf_size=32,
        matvec_repeats=1,
        use_single_precision=True,
    )
    assert len(report["orders"]) == 1
    assert report["orders"][0] > 1.9
    assert max(report["rel_subset_error"]) < 1e-5
    assert all(err > 0 for err in report["errors"])


def test_public_single_layer_demo_scripts_run():
    single_layer_demo_main(geometry="square", n_points=1000, rank_or_tol=1e-5, leaf_size=50)
    single_layer_demo_main(geometry="cube", n_points=1000, rank_or_tol=1e-5, leaf_size=50, kappa=2.0)
