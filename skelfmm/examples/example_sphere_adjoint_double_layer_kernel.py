"""User-facing sphere example for double-layer and adjoint double-layer kernels.

This example builds two ``SkelFMM`` objects on the same sphere panel geometry:

- the Laplace double-layer kernel
- the Laplace adjoint double-layer kernel

It reports build/apply timings and checks the discrete adjoint identity

    <D (w * x), y> = <w * x, D* y>

where ``w`` are the panel areas.

Run with:

    python -m skelfmm.examples.example_sphere_adjoint_double_layer_kernel
"""

import numpy as np

from skelfmm import SkelFMM, operators

from .util.geometry_utils import make_icosphere_mesh, triangle_panel_geometry


N_PANELS = 1_310_720
TOL = 1e-4
LEAF_SIZE = 200
MATVEC_REPEATS = 3


def main(
    *,
    n_panels=N_PANELS,
    tol=TOL,
    leaf_size=LEAF_SIZE,
    matvec_repeats=MATVEC_REPEATS,
    seed=0,
):
    vertices, faces = make_icosphere_mesh(n_panels=n_panels, radius=1.0)
    points, normals, area = triangle_panel_geometry(vertices, faces)

    fmm_double = SkelFMM(
        points,
        operators.LAPLACE_DOUBLE_LAYER_3D,
        geometry=normals,
        tol=tol,
        leaf_size=leaf_size,
    )
    fmm_adjoint = SkelFMM(
        points,
        operators.LAPLACE_ADJOINT_DOUBLE_LAYER_3D,
        geometry=normals,
        tol=tol,
        leaf_size=leaf_size,
    )

    random = np.random.RandomState(seed)
    x = random.randn(points.shape[0])
    y = random.randn(points.shape[0])

    # The quadrature weights enter through the source vector, while the panel
    # normals are passed separately as the geometry attached to the kernel.
    weighted_x = area * x

    double_layer_out = fmm_double.apply(weighted_x)
    adjoint_out = fmm_adjoint.apply(y)
    double_layer_benchmark = fmm_double.benchmark_matvec(
        nrepeat=matvec_repeats,
        seed=0,
    )
    adjoint_benchmark = fmm_adjoint.benchmark_matvec(nrepeat=matvec_repeats, seed=1)

    lhs = np.dot(double_layer_out, y)
    rhs = np.dot(weighted_x, adjoint_out)
    adjoint_relerr = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)

    print("\tLaplace sphere double-layer and adjoint double-layer")
    print(f"\tN={points.shape[0]}  tol={tol:.1e}  leaf_size={leaf_size}")
    print(
        f"\tTbuild_double_s={fmm_double.build_time_seconds:.2f}  "
        f"rank_max_double={fmm_double.rank_max}  "
        f"Tapply_double_s={double_layer_benchmark['min_seconds']:.3f}"
    )
    print(
        f"\tTbuild_adjoint_s={fmm_adjoint.build_time_seconds:.2f}  "
        f"rank_max_adjoint={fmm_adjoint.rank_max}  "
        f"Tapply_adjoint_s={adjoint_benchmark['min_seconds']:.3f}"
    )
    print(f"\tadjoint_relerr={adjoint_relerr:.3e}")

    return {
        "points": points,
        "normals": normals,
        "area": area,
        "fmm_double": fmm_double,
        "fmm_adjoint": fmm_adjoint,
        "build_double_s": fmm_double.build_time_seconds,
        "build_adjoint_s": fmm_adjoint.build_time_seconds,
        "tapply_double_s": double_layer_benchmark["min_seconds"],
        "tapply_adjoint_s": adjoint_benchmark["min_seconds"],
        "rank_max_double": fmm_double.rank_max,
        "rank_max_adjoint": fmm_adjoint.rank_max,
        "adjoint_relerr": float(adjoint_relerr),
    }


if __name__ == "__main__":
    main()
