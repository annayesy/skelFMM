from dataclasses import dataclass
from functools import partial

from . import util


def _call_batched_func(name, *args, **kwargs):
    from . import util_batched

    return getattr(util_batched, name)(*args, **kwargs)


def _lazy_batched_func(name):
    return partial(_call_batched_func, name)


@dataclass
class KernelOperator:
    """Stateless kernel/operator interface for dense and batched evaluation."""

    name: str = "custom"
    ndim: int = -1
    numpy_func: object = None
    torch_func: object = None
    keops_func: object = None
    proxy_surface: str = "cube"
    geometry_side: str | None = None
    is_symmetric: bool = False

    def __post_init__(self):
        if self.geometry_side not in {None, "source", "target"}:
            raise ValueError("geometry_side must be None, 'source', or 'target'")

    @property
    def requires_geometry(self):
        return self.geometry_side is not None

    @property
    def has_keops(self):
        return self.keops_func is not None

    def _geom_kwargs(self, geom_trg=None, geom_src=None):
        if self.geometry_side == "source":
            return {"src_geom": geom_src}
        if self.geometry_side == "target":
            return {"trg_geom": geom_trg}
        return {}

    def numpy_matrix(
        self,
        x_trg,
        x_src,
        *,
        geom_trg=None,
        geom_src=None,
        param=0.0,
        assume_not_self_interaction=False,
    ):
        if self.numpy_func is None:
            raise NotImplementedError
        return self.numpy_func(
            x_trg,
            x_src,
            param=param,
            assume_not_self_interaction=assume_not_self_interaction,
            **self._geom_kwargs(geom_trg, geom_src),
        )

    def torch_apply(
        self,
        q,
        x_trg,
        x_src,
        *,
        geom_trg=None,
        geom_src=None,
        param=0.0,
        self_interaction=False,
    ):
        if self.torch_func is None:
            raise NotImplementedError
        return self.torch_func(
            q,
            x_trg,
            x_src,
            param,
            self_interaction,
            **self._geom_kwargs(geom_trg, geom_src),
        )

    def keops_apply(
        self,
        q,
        x_trg,
        x_src,
        *,
        geom_trg=None,
        geom_src=None,
        param=0.0,
        self_interaction=False,
    ):
        if self.keops_func is None:
            return None

        geom_kwargs = self._geom_kwargs(geom_trg, geom_src)
        if param == 0.0:
            return self.keops_func(q, x_trg, x_src, self_interaction, **geom_kwargs)
        return self.keops_func(q, x_trg, x_src, param, self_interaction, **geom_kwargs)


class PointKernelOperator(KernelOperator):
    """Operator adapter for point kernels that do not use extra geometry."""

    def __init__(
        self,
        name,
        ndim,
        numpy_func,
        torch_func,
        keops_func=None,
        proxy_surface="cube",
        is_symmetric=True,
    ):
        super().__init__(
            name=name,
            ndim=ndim,
            numpy_func=numpy_func,
            torch_func=torch_func,
            keops_func=keops_func,
            proxy_surface=proxy_surface,
            is_symmetric=is_symmetric,
        )


class GeometryAwareKernelOperator(KernelOperator):
    """Operator for kernels that carry one aligned source or target geometry."""

    def __init__(
        self,
        name,
        ndim,
        numpy_func,
        torch_func,
        keops_func,
        *,
        geometry_side,
    ):
        super().__init__(
            name=name,
            ndim=ndim,
            numpy_func=numpy_func,
            torch_func=torch_func,
            keops_func=keops_func,
            proxy_surface="sphere",
            geometry_side=geometry_side,
        )


LAPLACE_2D = PointKernelOperator(
    name="laplace_2d",
    ndim=2,
    numpy_func=util.laplace_2d,
    torch_func=_lazy_batched_func("laplace_2d"),
    keops_func=_lazy_batched_func("laplace_2d_lazy"),
)
LAPLACE_3D = PointKernelOperator(
    name="laplace_3d",
    ndim=3,
    numpy_func=util.laplace_3d,
    torch_func=_lazy_batched_func("laplace_3d"),
    keops_func=_lazy_batched_func("laplace_3d_lazy"),
)
HELMHOLTZ_2D = PointKernelOperator(
    name="helmholtz_2d",
    ndim=2,
    numpy_func=util.helmholtz_2d,
    torch_func=_lazy_batched_func("helmholtz_2d"),
)
HELMHOLTZ_3D = PointKernelOperator(
    name="helmholtz_3d",
    ndim=3,
    numpy_func=util.helmholtz_3d,
    torch_func=_lazy_batched_func("helmholtz_3d"),
    keops_func=_lazy_batched_func("helmholtz_3d_lazy"),
)
LAPLACE_DOUBLE_LAYER_3D = GeometryAwareKernelOperator(
    name="laplace_double_layer_3d",
    ndim=3,
    numpy_func=util.laplace_3d_double_layer,
    torch_func=_lazy_batched_func("laplace_3d_double_layer"),
    keops_func=_lazy_batched_func("laplace_3d_double_layer_lazy"),
    geometry_side="source",
)
LAPLACE_ADJOINT_DOUBLE_LAYER_3D = GeometryAwareKernelOperator(
    name="laplace_adjoint_double_layer_3d",
    ndim=3,
    numpy_func=util.laplace_3d_adjoint_double_layer,
    torch_func=_lazy_batched_func("laplace_3d_adjoint_double_layer"),
    keops_func=_lazy_batched_func("laplace_3d_adjoint_double_layer_lazy"),
    geometry_side="target",
)
HELMHOLTZ_DOUBLE_LAYER_3D = GeometryAwareKernelOperator(
    name="helmholtz_double_layer_3d",
    ndim=3,
    numpy_func=util.helmholtz_3d_double_layer,
    torch_func=_lazy_batched_func("helmholtz_3d_double_layer"),
    keops_func=_lazy_batched_func("helmholtz_3d_double_layer_lazy"),
    geometry_side="source",
)
HELMHOLTZ_ADJOINT_DOUBLE_LAYER_3D = GeometryAwareKernelOperator(
    name="helmholtz_adjoint_double_layer_3d",
    ndim=3,
    numpy_func=util.helmholtz_3d_adjoint_double_layer,
    torch_func=_lazy_batched_func("helmholtz_3d_adjoint_double_layer"),
    keops_func=_lazy_batched_func("helmholtz_3d_adjoint_double_layer_lazy"),
    geometry_side="target",
)


def _require_operator(operator):
    """Validate that an explicit `KernelOperator` was provided."""
    if not isinstance(operator, KernelOperator):
        raise TypeError(
            "Expected a KernelOperator instance. "
            "Use objects from skelfmm.operators such as operators.LAPLACE_3D."
        )
    return operator
