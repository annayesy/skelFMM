import numpy as np
from time import perf_counter

from skelfmm import SkelFMM
from skelfmm import operators
from .geometry_utils import make_point_geometry
from .sphere_kernel_utils import fmm_fast_apply_stats


def select_single_layer_operator(ndim: int, kappa: float):
    operator_family = "laplace" if kappa == 0.0 else "helmholtz"
    operator_table = {
        ("laplace", 2): operators.LAPLACE_2D,
        ("laplace", 3): operators.LAPLACE_3D,
        ("helmholtz", 2): operators.HELMHOLTZ_2D,
        ("helmholtz", 3): operators.HELMHOLTZ_3D,
    }
    try:
        operator = operator_table[(operator_family, ndim)]
    except KeyError as exc:
        raise ValueError(f"Unsupported dimension for single-layer operator selection: ndim={ndim}") from exc
    return operator_family, operator, float(kappa)


def run_single_layer_demo(
    *,
    geometry: str,
    n_points: int = 1_000_000,
    rank_or_tol: float = 1e-5,
    leaf_size: int = 200,
    kappa: float = 0.0,
    use_single_precision: bool = True,
    ncheck: int = 50,
    matvec_repeats: int = 3,
):
    """
    Run a single-layer example over one of the built-in point geometries.
    """
    t0 = perf_counter()
    points = make_point_geometry(geometry, n_points)
    generation_time_seconds = perf_counter() - t0
    kernel, operator, param = select_single_layer_operator(points.shape[1], kappa)

    fmm_apply = SkelFMM(
        points,
        operator,
        geometry=None,
        tol=rank_or_tol,
        leaf_size=leaf_size,
        param=param,
        use_single_precision=use_single_precision,
    )

    rng = np.random.default_rng(0)
    x = rng.random(n_points)
    benchmark = fmm_apply.benchmark_matvec(nrepeat=matvec_repeats, seed=0)
    relerr_inf = fmm_apply.relerr_fmm_apply_check(x, ncheck=ncheck, seed=0)
    stats = fmm_fast_apply_stats(fmm_apply)

    return {
        "kernel": kernel,
        "geometry": geometry,
        "ndim": points.shape[1],
        "n_points": n_points,
        "kappa": param,
        "rank_or_tol": rank_or_tol,
        "leaf_size": leaf_size,
        "generation_time_seconds": generation_time_seconds,
        "build_time_seconds": stats["build_time_seconds"],
        "rank_max": stats["rank_max"],
        "matvec_repeats": matvec_repeats,
        "matvec_avg_seconds": benchmark["avg_seconds"],
        "matvec_min_seconds": benchmark["min_seconds"],
        "matvec_max_seconds": benchmark["max_seconds"],
        "backend": stats["backend"],
        "device": stats["device"],
        "host_storage_bytes": stats["host_storage_bytes"],
        "host_build_storage_bytes": stats["host_build_storage_bytes"],
        "device_storage_bytes": stats["device_storage_bytes"],
        "total_storage_bytes": stats["total_storage_bytes"],
        "nlevels": stats["nlevels"],
        "nboxes": stats["nboxes"],
        "nleaves": stats["nleaves"],
        "relerr_inf": relerr_inf,
        "points": points,
    }


def format_single_layer_tapply(report):
    if "cuda" in str(report["device"]):
        return f"Tapply_ms={1e3 * report['matvec_min_seconds']:.1f}"
    return f"Tapply_s={report['matvec_min_seconds']:.3e}"


def print_single_layer_report(
    report,
    *,
    title=None,
    use_single_precision=True,
):
    kernel_label = "Laplace" if report["kappa"] == 0.0 else "Helmholtz"
    default_title = f"{kernel_label} single-layer on geometry={report['geometry']!r}"
    print(f"\t{title or default_title}")
    print(f"\tN={report['n_points']}  ndim={report['ndim']}  tol={report['rank_or_tol']:.1e}")
    if report["kappa"] != 0.0:
        print(f"\tkappa={report['kappa']}")
    summary_parts = [
        f"Tgen_s={report['generation_time_seconds']:.2f}",
        f"Tbuild_s={report['build_time_seconds']:.2f}",
        format_single_layer_tapply(report),
        f"mem_gb={report['total_storage_bytes'] / 1e9:.3f}",
    ]
    if "leaf_size" in report:
        summary_parts.insert(1, f"leaf_size={report['leaf_size']}")
    if "rank_max" in report:
        summary_parts.insert(-2, f"rank_max={report['rank_max']}")
    print("\t" + "  ".join(summary_parts))
    precision_label = "single" if use_single_precision else "double"
    print(f"\tbackend={report['backend']}  device={report['device']}  precision={precision_label}")
    print(f"\trelerr_inf={report['relerr_inf']:.3e}")
