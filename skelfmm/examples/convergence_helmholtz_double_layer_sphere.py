"""Helmholtz double-layer sphere convergence example."""

from skelfmm import operators

from .util.report_utils import mesh_convergence_table
from .util.sphere_kernel_utils import (
    benchmark_sphere_apply,
    estimate_convergence as estimate_panel_convergence,
    fast_self_apply_error as estimate_self_apply_error,
    print_sphere_kernel_summary,
)


def dense_double_layer_potential(targets, points, normals, weights, density, param):
    return operators.HELMHOLTZ_DOUBLE_LAYER_3D.numpy_matrix(
        targets,
        points,
        geom_src=normals,
        param=param,
    ) @ (weights * density)


def fast_self_apply_error(
    n_points=256,
    kappa=30.0,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    ncheck=64,
    seed=0,
    use_single_precision=True,
):
    return estimate_self_apply_error(
        operators.HELMHOLTZ_DOUBLE_LAYER_3D,
        n_points=n_points,
        tol=tol,
        rank_or_tol=rank_or_tol,
        leaf_size=leaf_size,
        ncheck=ncheck,
        seed=seed,
        use_single_precision=use_single_precision,
        param=kappa,
        use_normals=True,
    )


def benchmark_sphere(
    n_panels=1_310_720,
    kappa=30.0,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    matvec_repeats=3,
    use_single_precision=True,
):
    return benchmark_sphere_apply(
        operators.HELMHOLTZ_DOUBLE_LAYER_3D,
        n_panels=n_panels,
        tol=tol,
        rank_or_tol=rank_or_tol,
        leaf_size=leaf_size,
        matvec_repeats=matvec_repeats,
        use_single_precision=use_single_precision,
        param=kappa,
        use_normals=True,
    )


def estimate_convergence(
    n_points_list=None,
    n_panels_list=(1280, 5120, 20480, 81920, 327680, 1_310_720),
    targets=None,
    kappa=30.0,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    matvec_repeats=3,
    use_single_precision=True,
    verbose=False,
):
    report = estimate_panel_convergence(
        operators.HELMHOLTZ_DOUBLE_LAYER_3D,
        dense_double_layer_potential,
        n_points_list=n_points_list,
        n_panels_list=n_panels_list,
        targets=targets,
        tol=tol,
        rank_or_tol=rank_or_tol,
        leaf_size=leaf_size,
        matvec_repeats=matvec_repeats,
        use_single_precision=use_single_precision,
        param=kappa,
        use_normals=True,
        verbose=verbose,
        verbose_suffix=f"  kappa={kappa}",
    )
    report["kappa"] = kappa
    return report


def main():
    kappa = 30.0
    report = estimate_convergence(kappa=kappa, leaf_size=200, verbose=True)
    bench = benchmark_sphere(kappa=kappa, leaf_size=200)
    print_sphere_kernel_summary(
        "Helmholtz double-layer kernel on the sphere",
        report,
        bench,
        fast_self_apply_error(kappa=kappa, leaf_size=200),
        kappa=kappa,
        note="relerr is the off-surface panel discretization error; fmm_relerr is the subset dense-vs-fast infinity-norm error.",
    )
    for line in mesh_convergence_table(report).splitlines():
        print(f"\t{line}")


if __name__ == "__main__":
    main()
