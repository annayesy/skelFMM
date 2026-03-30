import contextlib
import io
from time import perf_counter

import numpy as np
import torch

from simpletree import BalancedTree

from .recursive_skel import RecursiveSkeletonization


def _estimate_host_storage_bytes_from_fmm(fmm):
    total = 0
    arrays = (
        fmm.XX_list,
        fmm.geom_list,
        fmm.idx_list,
        fmm.proj_list,
        fmm.proj_list_tree,
        fmm.bs_list,
        fmm.rank_list,
        fmm.tree_boxes,
        fmm.tree_lev_sep,
    )
    for arr in arrays:
        if arr is not None:
            total += arr.nbytes
    return int(total)


def _batched_tensor_dtype(x, real_dtype):
    if np.iscomplexobj(x):
        return torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return real_dtype


def warmup_cuda_fmm_apply(apply, x):
    from . import util_batched

    device = getattr(apply, "device", None)
    if not (isinstance(device, torch.device) and device.type == "cuda"):
        return False

    util_batched.warmup_cuda_memory_pool(device, verbose=False)
    apply.apply(x)
    synchronize = getattr(apply, "_synchronize", None)
    if synchronize is not None:
        synchronize()
    else:
        torch.cuda.synchronize(device)
    return True


class SkelFMM:
    """Build and apply a fast operator through the public batched interface."""

    def __init__(
        self,
        points,
        operator,
        geometry=None,
        tol=1e-5,
        rank_or_tol=None,
        leaf_size=200,
        use_single_precision=False,
        param=0.0,
        quiet=True,
        keep_host_fmm=False,
        defer_build=False,
        npoints_max=500,
        p="auto",
        max_bs=-1,
        max_rank=-1,
    ):
        self.points = points
        self.operator = operator
        self.geometry = geometry
        self.param = param
        self.rank_or_tol = tol if rank_or_tol is None else rank_or_tol
        self.leaf_size = leaf_size
        self.tree = None
        self.nlevels = 0
        self.nboxes = 0
        self.nleaves = 0
        self._quiet = quiet
        self._keep_host_fmm = keep_host_fmm
        self._use_single_precision = use_single_precision
        self._build_npoints_max = npoints_max
        self._build_p = p
        self._max_bs_limit = max_bs
        self._max_rank_limit = max_rank
        self._is_built = False
        self._lazy_build_notice_printed = False

        self.fmm = None
        self.batched = None
        self.device = None
        self.kernel_backend = None
        self.tree_time_seconds = 0.0
        self.skel_time_seconds = 0.0
        self.setup_time_seconds = 0.0
        self.build_time_seconds = 0.0
        self.rank_max = 0
        self.host_build_storage_bytes = 0
        self.host_storage_bytes = 0
        self.device_storage_bytes = 0
        self.total_storage_bytes = 0
        self.nbytes = 0

        if not defer_build:
            self.build()

    def _estimate_host_storage_bytes(self):
        if self.fmm is None:
            return 0
        return _estimate_host_storage_bytes_from_fmm(self.fmm)

    def _compute_rank_max(self):
        if self.fmm is None:
            return 0
        rank_list = np.asarray(self.fmm.rank_list, dtype=int)
        positive = rank_list[rank_list > 0]
        if positive.size == 0:
            return 0
        return int(np.max(positive))

    def _synchronize(self):
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def build(self, *, npoints_max=None, p=None, verbose=False):
        if self._is_built:
            return self

        if npoints_max is None:
            npoints_max = self._build_npoints_max
        if p is None:
            p = self._build_p

        build_t0 = perf_counter()
        stream_ctx = contextlib.nullcontext()
        if self._quiet:
            stream_ctx = contextlib.ExitStack()
        with stream_ctx as stack:
            if self._quiet:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            tree_t0 = perf_counter()
            self.tree = BalancedTree(self.points, leaf_size=self.leaf_size)
            self.tree_time_seconds = perf_counter() - tree_t0
            self.nlevels = self.tree.nlevels
            self.nboxes = self.tree.nboxes
            self.nleaves = self.tree.nleaves
            self.fmm = RecursiveSkeletonization(
                self.tree,
                self.operator,
                kappa=self.param,
                tol=self.rank_or_tol,
                max_bs=self._max_bs_limit,
                max_rank=self._max_rank_limit,
                geometry=self.geometry,
            )
            skel_t0 = perf_counter()
            self.fmm.skel_tree(npoints_max=npoints_max, p=p, verbose=verbose)
            self.skel_time_seconds = perf_counter() - skel_t0

            setup_t0 = perf_counter()
            self.fmm.setup_lists(verbose=verbose)

            from .batched_fmm import SkelFMMBatched

            self.batched = SkelFMMBatched(self.tree, self.operator, kappa=self.param)
            self.batched.build_from_copy(self.fmm, use_single_precision=self._use_single_precision)
            self.setup_time_seconds = perf_counter() - setup_t0
        self.build_time_seconds = perf_counter() - build_t0

        self.device = self.batched.device
        self.kernel_backend = self.batched.kernel_backend
        host_build_storage_bytes = self._estimate_host_storage_bytes()
        self.rank_max = self._compute_rank_max()
        if self._keep_host_fmm:
            host_storage_bytes = host_build_storage_bytes
        else:
            self.fmm = None
            host_storage_bytes = 0
        device_storage_bytes = int(self.batched.nbytes_proj() + self.batched.nbytes_lists())
        total_storage_bytes = host_storage_bytes + device_storage_bytes
        self.host_build_storage_bytes = host_build_storage_bytes
        self.host_storage_bytes = host_storage_bytes
        self.device_storage_bytes = device_storage_bytes
        self.total_storage_bytes = total_storage_bytes
        self.nbytes = total_storage_bytes
        self._is_built = True
        return self

    def _ensure_built(self, *, announce=False):
        if getattr(self, "_is_built", True):
            return
        if announce and not self._lazy_build_notice_printed:
            print("\tBuilding SkelFMM on first matvec; subsequent matvecs will be fast.")
            self._lazy_build_notice_printed = True
        self.build()

    def apply(self, x):
        self._ensure_built(announce=True)
        x_np = np.asarray(x)
        if x_np.ndim != 1:
            raise ValueError("apply expects a 1D vector")

        x_t = torch.as_tensor(x_np, dtype=_batched_tensor_dtype(x_np, self.batched.XX_list.dtype))
        y_t = self.batched.matvec(x_t, verbose=False)
        return y_t.detach().cpu().numpy()

    def matvec(self, x):
        return self.apply(x)

    def relerr_fmm_apply_check(self, x, *, ncheck=64, seed=0, indices=None):
        """
        Return a relative ``ell_infinity`` subset apply error for this fast apply.

        This compares the fast matvec against a dense operator evaluation on a
        subset of targets:

            ||y_dense - y_fmm||_inf / ||y_dense||_inf
        """
        self._ensure_built()
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("relerr_fmm_apply_check expects a 1D source vector")
        if x.shape[0] != self.points.shape[0]:
            raise ValueError("Source vector length must match the number of points")

        if indices is None:
            rng = np.random.default_rng(seed)
            indices = rng.choice(self.points.shape[0], size=min(int(ncheck), self.points.shape[0]), replace=False)
        else:
            indices = np.asarray(indices, dtype=int)

        geom_src = None
        geom_trg = None
        if self.operator.geometry_side == "source":
            geom_src = self.geometry
        elif self.operator.geometry_side == "target":
            geom_trg = self.geometry[indices]

        y_fast = self.apply(x)
        y_true = self.operator.numpy_matrix(
            self.points[indices],
            self.points,
            geom_trg=geom_trg,
            geom_src=geom_src,
            param=self.param,
        ) @ x

        denom = max(np.linalg.norm(y_true, ord=np.inf), 1e-30)
        return np.linalg.norm(y_true - y_fast[indices], ord=np.inf) / denom

    def benchmark_matvec(self, *, nrepeat=3, warmup=1, seed=0):
        self._ensure_built()
        random = np.random.RandomState(seed)
        x = random.randn(self.points.shape[0])

        warmup_cuda_fmm_apply(self, x)
        times = []
        for _ in range(max(warmup, 0)):
            self.apply(x)
            self._synchronize()

        for _ in range(max(nrepeat, 1)):
            self._synchronize()
            t0 = perf_counter()
            self.apply(x)
            self._synchronize()
            times.append(perf_counter() - t0)

        return {
            "times_seconds": tuple(times),
            "avg_seconds": float(np.mean(times)),
            "min_seconds": float(np.min(times)),
            "max_seconds": float(np.max(times)),
        }
