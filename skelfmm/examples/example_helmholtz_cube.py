"""Self-contained Helmholtz example on a cube using ``SkelFMM``.

Run with:

    python -m skelfmm.examples.example_helmholtz_cube
"""

import numpy as np
from simpletree import get_point_dist

from skelfmm import SkelFMM, operators


N_POINTS = 1_000_000
TOL = 1e-5
KAPPA = 5.0
LEAF_SIZE = 320
MATVEC_REPEATS = 3
NCHECK = 64


def main():
    points = get_point_dist(N_POINTS, "cube")
    fmm = SkelFMM(
        points,
        operators.HELMHOLTZ_3D,
        tol=TOL,
        leaf_size=LEAF_SIZE,
        param=KAPPA,
        use_single_precision=True,
    )

    random = np.random.RandomState(0)
    x = random.randn(points.shape[0])
    y = fmm.apply(x)
    benchmark = fmm.benchmark_matvec(nrepeat=MATVEC_REPEATS, seed=0)

    relerr = fmm.relerr_fmm_apply_check(x, ncheck=NCHECK, seed=0)
    tapply = benchmark["min_seconds"]
    if getattr(fmm.device, "type", None) == "cuda":
        tapply_label = f"Tapply_ms={1e3 * tapply:.1f}"
    else:
        tapply_label = f"Tapply_s={tapply:.3f}"

    print("\tHelmholtz single-layer on a cube")
    print(f"\tN={points.shape[0]}  tol={TOL:.1e}  kappa={KAPPA:.1f}  leaf_size={LEAF_SIZE}")
    print(
        f"\tTbuild_s={fmm.build_time_seconds:.2f}  "
        f"{tapply_label}  rank_max={fmm.rank_max}"
    )
    print(f"\tfmm_relerr={relerr:.3e}")

    return {
        "points": points,
        "fmm": fmm,
        "output": y,
        "kappa": KAPPA,
        "tbuild": fmm.build_time_seconds,
        "tapply_min": benchmark["min_seconds"],
        "rank_max": fmm.rank_max,
        "fmm_relerr": float(relerr),
    }


if __name__ == "__main__":
    main()
