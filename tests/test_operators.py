import numpy as np
import pytest
import torch

import skelfmm
from skelfmm import operators


def test_package_lazy_batched_import_and_bad_attr():
    assert skelfmm.SkelFMMBatched is not None
    with pytest.raises(AttributeError):
        _ = skelfmm.not_a_real_symbol


def test_kernel_operator_base_methods_and_metadata():
    base = operators.KernelOperator()

    with pytest.raises(NotImplementedError):
        base.numpy_matrix(np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(NotImplementedError):
        base.torch_apply(
            torch.ones(2),
            torch.zeros((2, 2)),
            torch.zeros((2, 2)),
        )

    assert base.keops_apply(None, None, None) is None
    assert base.has_keops is False
    assert base.proxy_surface == "cube"
    assert base.geometry_side is None
    assert base.is_symmetric is False


def test_point_kernel_operator_keops_and_has_keops():
    def numpy_func(x_trg, x_src, param, assume_not_self_interaction=False):
        return np.full((x_trg.shape[0], x_src.shape[0]), 1.0 + param)

    def torch_func(q, x_trg, x_src, param, self_interaction):
        return q + param + float(self_interaction)

    def keops_func_zero(q, x_trg, x_src, self_interaction):
        return q + 10.0 + float(self_interaction)

    op_zero = operators.PointKernelOperator(
        name="toy_zero",
        ndim=2,
        numpy_func=numpy_func,
        torch_func=torch_func,
        keops_func=keops_func_zero,
    )
    q = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = op_zero.keops_apply(q, None, None, param=0.0, self_interaction=True)
    assert torch.allclose(out, torch.tensor([12.0, 13.0], dtype=torch.float64))
    assert op_zero.has_keops is True

    def keops_func_param(q, x_trg, x_src, param, self_interaction):
        return q + param + 20.0 + float(self_interaction)

    op_param = operators.PointKernelOperator(
        name="toy_param",
        ndim=2,
        numpy_func=numpy_func,
        torch_func=torch_func,
        keops_func=keops_func_param,
    )
    out = op_param.keops_apply(q, None, None, param=3.0, self_interaction=False)
    assert torch.allclose(out, torch.tensor([24.0, 25.0], dtype=torch.float64))

    op_no_keops = operators.PointKernelOperator(
        name="toy_none",
        ndim=2,
        numpy_func=numpy_func,
        torch_func=torch_func,
    )
    assert op_no_keops.keops_apply(q, None, None, param=0.0, self_interaction=False) is None
    assert op_no_keops.has_keops is False
    assert op_no_keops.is_symmetric is True
    assert np.allclose(op_no_keops.numpy_matrix(np.zeros((2, 2)), np.zeros((3, 2)), param=2.0), 3.0)
    torch_out = op_no_keops.torch_apply(q, None, None, param=2.0, self_interaction=True)
    assert torch.allclose(torch_out, torch.tensor([4.0, 5.0], dtype=torch.float64))


def test_double_layer_operator_metadata():
    dl = operators.LAPLACE_DOUBLE_LAYER_3D
    adl = operators.LAPLACE_ADJOINT_DOUBLE_LAYER_3D
    hdl = operators.HELMHOLTZ_DOUBLE_LAYER_3D
    hadl = operators.HELMHOLTZ_ADJOINT_DOUBLE_LAYER_3D

    assert dl.proxy_surface == "sphere"
    assert adl.proxy_surface == "sphere"
    assert dl.is_symmetric is False
    assert adl.is_symmetric is False
    assert hdl.has_keops is True
    assert hadl.has_keops is True
    assert hdl.proxy_surface == "sphere"
    assert hadl.proxy_surface == "sphere"
    assert hdl.geometry_side == "source"
    assert hadl.geometry_side == "target"


def test_helmholtz_adjoint_double_layer_satisfies_bilinear_identity():
    rng = np.random.default_rng(7)
    random = np.random.RandomState(7)
    src = rng.normal(size=(48, 3))
    trg = rng.normal(size=(32, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    trg /= np.linalg.norm(trg, axis=1, keepdims=True)

    x = random.randn(src.shape[0]) + 1j * random.randn(src.shape[0])
    y = random.randn(trg.shape[0]) + 1j * random.randn(trg.shape[0])

    K = operators.HELMHOLTZ_DOUBLE_LAYER_3D.numpy_matrix(trg, src, geom_src=src, param=1.7)
    K_adj = operators.HELMHOLTZ_ADJOINT_DOUBLE_LAYER_3D.numpy_matrix(src, trg, geom_trg=src, param=1.7)

    lhs = np.dot(K @ x, y)
    rhs = np.dot(x, K_adj @ y)
    assert np.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_laplace_adjoint_double_layer_satisfies_bilinear_identity():
    rng = np.random.default_rng(11)
    random = np.random.RandomState(11)
    src = rng.normal(size=(48, 3))
    trg = rng.normal(size=(32, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    trg /= np.linalg.norm(trg, axis=1, keepdims=True)

    x = random.randn(src.shape[0])
    y = random.randn(trg.shape[0])

    K = operators.LAPLACE_DOUBLE_LAYER_3D.numpy_matrix(trg, src, geom_src=src)
    K_adj = operators.LAPLACE_ADJOINT_DOUBLE_LAYER_3D.numpy_matrix(src, trg, geom_trg=src)

    lhs = np.dot(K @ x, y)
    rhs = np.dot(x, K_adj @ y)
    assert np.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_internal_operator_requirement():
    assert operators._require_operator(operators.LAPLACE_3D) is operators.LAPLACE_3D

    with pytest.raises(TypeError):
        operators._require_operator(operators.util.laplace_3d)
    with pytest.raises(TypeError):
        operators._require_operator(123)
