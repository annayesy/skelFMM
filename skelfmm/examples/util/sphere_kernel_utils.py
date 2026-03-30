from time import perf_counter

import numpy as np

from skelfmm import SkelFMM, warmup_cuda_fmm_apply

from .geometry_utils import make_icosphere_mesh, make_sphere_mesh, triangle_panel_geometry


def fmm_fast_apply_stats(fmm_apply):
    """Return the example-facing summary statistics for one fast apply object."""
    return {
        "operator": fmm_apply.operator.name,
        "backend": fmm_apply.kernel_backend,
        "device": str(fmm_apply.device),
        "rank_or_tol": fmm_apply.rank_or_tol,
        "leaf_size": fmm_apply.leaf_size,
        "build_time_seconds": fmm_apply.build_time_seconds,
        "skel_time_seconds": fmm_apply.skel_time_seconds,
        "setup_time_seconds": fmm_apply.setup_time_seconds,
        "host_build_storage_bytes": fmm_apply.host_build_storage_bytes,
        "host_storage_bytes": fmm_apply.host_storage_bytes,
        "device_storage_bytes": fmm_apply.device_storage_bytes,
        "total_storage_bytes": fmm_apply.total_storage_bytes,
        "nbytes": fmm_apply.nbytes,
        "nlevels": fmm_apply.nlevels,
        "nboxes": fmm_apply.nboxes,
        "nleaves": fmm_apply.nleaves,
        "rank_max": fmm_apply.rank_max,
    }


def smooth_density(points):
    x, y, z = points.T
    return 1.0 + 0.3 * x - 0.2 * y + 0.1 * z


def make_panel_geometry(*, n_points=None, n_panels=None):
    """Return row-major panel centers, outward normals, and panel areas."""
    if (n_points is None) == (n_panels is None):
        raise ValueError("Exactly one of n_points or n_panels must be provided")
    if n_panels is not None:
        vertices, faces = make_icosphere_mesh(n_panels=n_panels, radius=1.0)
    else:
        vertices, faces = make_sphere_mesh(n_points=n_points, radius=1.0)
    return triangle_panel_geometry(vertices, faces)


def _panel_density_data(*, n_points=None, n_panels=None):
    """
    Return the panel geometry together with the source vector used in the FMM.

    For both single- and double-layer panel discretizations, the quadrature
    weights enter through the source vector ``weighted_density = weights * density``.
    The double-layer examples additionally pass the outward panel normals as the
    geometry attached to the source points.
    """
    points, normals, weights = make_panel_geometry(n_points=n_points, n_panels=n_panels)
    density = smooth_density(points)
    weighted_density = weights * density
    return points, normals, weights, density, weighted_density


def _mesh_iterable(n_points_list=None, n_panels_list=None):
    if n_points_list is not None:
        return [("points", int(n_points)) for n_points in n_points_list]
    if n_panels_list is not None:
        return [("panels", int(n_panels)) for n_panels in n_panels_list]
    raise ValueError("Either n_panels_list or n_points_list must be provided")


def build_fast_apply(
    points,
    operator,
    *,
    normals=None,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    use_single_precision=True,
    param=0.0,
):
    """Build a fast apply object for a panel-based sphere discretization."""
    geometry = normals if normals is not None else None
    return SkelFMM(
        points,
        operator,
        geometry=geometry,
        tol=tol,
        rank_or_tol=tol if rank_or_tol is None else rank_or_tol,
        leaf_size=leaf_size,
        param=param,
        use_single_precision=use_single_precision,
    )


def fast_self_apply_error(
    operator,
    *,
    n_points,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    ncheck=64,
    seed=0,
    use_single_precision=True,
    param=0.0,
    use_normals=False,
):
    """Return the subset dense-vs-fast infinity-norm apply error on a sphere panel discretization."""
    points, normals, _weights, _density, weighted_density = _panel_density_data(n_points=n_points)

    fmm_apply = build_fast_apply(
        points,
        operator,
        normals=normals if use_normals else None,
        tol=tol,
        rank_or_tol=rank_or_tol,
        leaf_size=leaf_size,
        use_single_precision=use_single_precision,
        param=param,
    )
    return fmm_apply.relerr_fmm_apply_check(weighted_density, ncheck=ncheck, seed=seed)


def benchmark_sphere_apply(
    operator,
    *,
    n_panels,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    matvec_repeats=3,
    use_single_precision=True,
    param=0.0,
    use_normals=False,
):
    """Benchmark one FMM apply on the sphere panel hierarchy."""
    t0 = perf_counter()
    points, normals, weights, density, weighted_density = _panel_density_data(n_panels=n_panels)
    generation_time_seconds = perf_counter() - t0

    fmm_apply = build_fast_apply(
        points,
        operator,
        normals=normals if use_normals else None,
        tol=tol,
        rank_or_tol=rank_or_tol,
        leaf_size=leaf_size,
        use_single_precision=use_single_precision,
        param=param,
    )
    benchmark = fmm_apply.benchmark_matvec(nrepeat=matvec_repeats, seed=0)
    fmm_stats = fmm_fast_apply_stats(fmm_apply)

    return {
        "n_panels": points.shape[0],
        "generation_time_seconds": generation_time_seconds,
        **fmm_stats,
        "matvec_repeats": matvec_repeats,
        "matvec_avg_seconds": benchmark["avg_seconds"],
        "matvec_min_seconds": benchmark["min_seconds"],
        "matvec_max_seconds": benchmark["max_seconds"],
    }


def estimate_convergence(
    operator,
    dense_potential,
    *,
    n_points_list=None,
    n_panels_list=None,
    targets=None,
    tol=1e-4,
    rank_or_tol=None,
    leaf_size=200,
    matvec_repeats=3,
    use_single_precision=True,
    param=0.0,
    use_normals=False,
    verbose=False,
    verbose_suffix="",
):
    """Run the sphere convergence study using the panel-weighted source vector."""
    if targets is None:
        targets = np.array([[0.2, -0.1, 0.3], [-0.35, 0.25, 0.15]])

    values = []
    panels = []
    h_values = []
    rank_max = []
    tbuild_seconds = []
    tapply_seconds = []
    storage_mb = []
    rel_subset_error = []
    devices = []
    labels = []

    iterable = _mesh_iterable(n_points_list=n_points_list, n_panels_list=n_panels_list)

    for mode, count in iterable:
        if verbose:
            label = "N" if mode == "panels" else "n_points"
            print(f"\t{label}={count}{verbose_suffix}")

        if mode == "panels":
            points, normals, weights, density, weighted_density = _panel_density_data(n_panels=count)
        else:
            points, normals, weights, density, weighted_density = _panel_density_data(n_points=count)
        values.append(dense_potential(targets, points, normals, weights, density, param))

        fmm_apply = build_fast_apply(
            points,
            operator,
            normals=normals if use_normals else None,
            tol=tol,
            rank_or_tol=rank_or_tol,
            leaf_size=leaf_size,
            use_single_precision=use_single_precision,
            param=param,
        )
        fmm_stats = fmm_fast_apply_stats(fmm_apply)
        matvec_stats = fmm_apply.benchmark_matvec(nrepeat=matvec_repeats, seed=0)
        panels.append(points.shape[0])
        h_values.append(np.sqrt(4.0 * np.pi / points.shape[0]))
        rank_max.append(fmm_stats["rank_max"])
        tbuild_seconds.append(fmm_stats["build_time_seconds"])
        tapply_seconds.append(matvec_stats["min_seconds"])
        storage_mb.append(fmm_stats["total_storage_bytes"] / 1e6)
        rel_subset_error.append(fmm_apply.relerr_fmm_apply_check(weighted_density, ncheck=64, seed=0))
        devices.append(fmm_stats["device"])
        labels.append(count)

    ref = values[-1]
    errors = [np.linalg.norm(value - ref) / np.linalg.norm(ref) for value in values[:-1]]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
        for i in range(len(errors) - 1)
    ]

    return {
        "targets": targets,
        "n_points": None if n_points_list is None else tuple(n_points_list),
        "n_panels_requested": None if n_panels_list is None else tuple(n_panels_list),
        "panels": tuple(panels),
        "h": tuple(h_values),
        "errors": tuple(errors),
        "rank_max": tuple(rank_max),
        "tbuild_seconds": tuple(tbuild_seconds),
        "tapply_seconds": tuple(tapply_seconds),
        "tapply_unit": "ms" if any("cuda" in str(device) for device in devices) else "s",
        "storage_mb": tuple(storage_mb),
        "rel_subset_error": tuple(rel_subset_error),
        "matvec_repeats": int(matvec_repeats),
        "orders": tuple(orders),
        "mesh_labels": tuple(labels),
    }


def format_benchmark_tapply(bench):
    if "cuda" in str(bench["device"]):
        return f"Tapply_ms={1e3 * bench['matvec_min_seconds']:.1f}"
    return f"Tapply_s={bench['matvec_min_seconds']:.3e}"


def print_sphere_kernel_summary(
    title,
    report,
    bench,
    self_relerr,
    *,
    note,
    kappa=None,
):
    print(f"\t{title}")
    if kappa is not None:
        print(f"\tkappa={kappa}")
    print(f"\tSelf-check: fmm_relerr={self_relerr:.3e}")
    print(
        f"\tBenchmark: N={bench['n_panels']}  Tgen_s={bench['generation_time_seconds']:.2f}  "
        f"leaf_size={bench['leaf_size']}  Tbuild_s={bench['build_time_seconds']:.2f}  "
        f"rank_max={bench['rank_max']}  {format_benchmark_tapply(bench)}  "
        f"mem_mb={bench['total_storage_bytes'] / 1e6:.1f}"
    )
    print(f"\ttol={bench['rank_or_tol']:.1e}  backend={bench['backend']}  device={bench['device']}")
    print(f"\tTable note: {note}")
