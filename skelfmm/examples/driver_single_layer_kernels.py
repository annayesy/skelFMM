"""Driver for single-layer kernels on the built-in point geometries."""

import argparse

from .util.geometry_utils import POINT_GEOMETRIES
from .util.single_layer_geometry_utils import print_single_layer_report, run_single_layer_demo


def main(
    *,
    geometry,
    n_points,
    rank_or_tol,
    leaf_size=200,
    kappa=0.0,
    matvec_repeats=3,
    use_single_precision=True,
):
    report = run_single_layer_demo(
        geometry=geometry,
        n_points=int(n_points),
        rank_or_tol=rank_or_tol,
        leaf_size=leaf_size,
        kappa=kappa,
        matvec_repeats=matvec_repeats,
        use_single_precision=use_single_precision,
    )

    print_single_layer_report(
        report,
        use_single_precision=use_single_precision,
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run a single-layer FMM example on one of the built-in point geometries. "
            "Use --kappa 0 for Laplace and a nonzero --kappa for Helmholtz. "
            "Single precision is the default."
        ),
        epilog=(
            f"Available geometries: {', '.join(POINT_GEOMETRIES)}\n\n"
            "Examples:\n"
            "  python -m skelfmm.examples.driver_single_layer_kernels --geom square --rank_or_tol 1e-5 --N 1000000\n"
            "  python -m skelfmm.examples.driver_single_layer_kernels --geom cube --rank_or_tol 1e-5 --N 1000000 --kappa 30.0"
        ),
    )
    parser.add_argument("--geom", required=True, choices=POINT_GEOMETRIES, help="Built-in geometry name.")
    parser.add_argument("--rank_or_tol", required=True, type=float, help="Requested compression tolerance for the FMM.")
    parser.add_argument("--N", required=True, type=int, help="Number of source/target points to generate.")
    parser.add_argument("--leaf_size", type=int, default=200, help="Leaf size for the balanced tree.")
    parser.add_argument("--kappa", type=float, default=0.0, help="Helmholtz wavenumber. Use 0 for Laplace.")
    parser.add_argument("--matvec_repeats", type=int, default=3, help="Number of fast matvecs used to average the timing.")
    parser.add_argument(
        "--double_precision",
        action="store_false",
        dest="use_single_precision",
        default=True,
        help="Use double precision on the batched path.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        geometry=args.geom,
        n_points=args.N,
        rank_or_tol=args.rank_or_tol,
        leaf_size=args.leaf_size,
        kappa=args.kappa,
        matvec_repeats=args.matvec_repeats,
        use_single_precision=args.use_single_precision,
    )
