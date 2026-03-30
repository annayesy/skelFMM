"""Public package interface for skelfmm."""

from ._pykeops_cuda import configure_pykeops_cuda_environment

configure_pykeops_cuda_environment()

from .skel_fmm import SkelFMM, warmup_cuda_fmm_apply
from .recursive_skel import RecursiveSkeletonization
from . import operators

__all__ = [
    "SkelFMM",
    "RecursiveSkeletonization",
    "SkelFMMBatched",
    "operators",
    "warmup_cuda_fmm_apply",
]


def __getattr__(name):
    if name == "SkelFMMBatched":
        from .batched_fmm import SkelFMMBatched

        return SkelFMMBatched
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
