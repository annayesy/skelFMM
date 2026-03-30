import torch
import subprocess

_WARMED_CUDA_DEVICES = set()

try:
    from pykeops.torch import LazyTensor
    PYKEOPS_AVAILABLE = True
except ImportError:
    LazyTensor = None
    PYKEOPS_AVAILABLE = False
    print("Pykeops not available")

def get_bytes_available(device_ord):
    """
    Returns the free memory available on a given CUDA device.

    Args:
        device_ord (int or torch.device): Device ordinal or torch.device object.

    Returns:
        float: Approximate free memory in bytes.
    """
    default_memory = 10 * 1e9  # 10GB fallback

    if isinstance(device_ord, torch.device) and device_ord.type == "cpu":
        return default_memory

    if not torch.cuda.is_available():
        return default_memory

    # Ensure device_ord is an integer (device index)
    if isinstance(device_ord, torch.device):
        if device_ord.type != 'cuda':
            raise ValueError(f"Invalid device type: {device_ord.type}. Only CUDA devices are supported.")
        device_ord = device_ord.index if device_ord.index is not None else 0

    free_memory, _total_memory = torch.cuda.mem_get_info(device_ord)
    return free_memory


def get_gpu_memory_usage_MB():
    """
    Query GPU memory usage with nvidia-smi and return per-device usage in MB.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(x) for x in result.stdout.strip().splitlines() if x.strip()]


def warmup_cuda_memory_pool(device, dtype=torch.float64, verbose=False):
    """
    Allocate and release a large CUDA tensor once to warm the allocator/cache.

    The released tensor remains in PyTorch's cached allocator, which can make
    subsequent batched FMM kernels run more consistently on some systems.
    """
    if not torch.cuda.is_available():
        return False

    if isinstance(device, torch.device):
        if device.type != "cuda":
            return False
        device_index = device.index if device.index is not None else 0
        device = torch.device("cuda", device_index)
    else:
        device_index = int(device)
        device = torch.device("cuda", device_index)

    if device_index in _WARMED_CUDA_DEVICES:
        return False

    total_memory = torch.cuda.get_device_properties(device).total_memory
    try:
        gpu_memory_mb = get_gpu_memory_usage_MB()
        mem_occupied = gpu_memory_mb[device_index] * int(1e6)
        free_memory = max(total_memory - mem_occupied, 0)
    except Exception:
        free_memory, total_memory = torch.cuda.mem_get_info(device)

    element_size = torch.empty((), dtype=dtype).element_size()

    target_bytes = max(int(0.9 * free_memory), free_memory - int(5e8))
    fallback_bytes = max(int(0.5 * target_bytes), 0)
    warmed = False

    for usable_bytes in (target_bytes, fallback_bytes):
        if usable_bytes <= element_size:
            continue

        n_max = int((usable_bytes / element_size) ** 0.5)
        if n_max <= 0:
            continue

        try:
            tmp = torch.rand(n_max, n_max, device=device, dtype=dtype)
            torch.cuda.synchronize(device)
            del tmp
            torch.cuda.synchronize(device)
            if verbose:
                print(
                    "\t Warmed CUDA allocator with %d x %d block (%5.2f GB free, %5.2f GB total)"
                    % (n_max, n_max, free_memory * 1e-9, total_memory * 1e-9)
                )
            warmed = True
            break
        except RuntimeError:
            continue

    _WARMED_CUDA_DEVICES.add(device_index)
    return warmed

def zero_row_bnd(B_tensor: torch.Tensor, row_bnds: torch.Tensor) -> torch.Tensor:
    b, m, n = B_tensor.shape
    device = B_tensor.device

    row_inds = torch.arange(m, device=device).reshape(1, m, 1)
    row_bnds = row_bnds.reshape(b, 1, 1)

    mask = row_inds < row_bnds
    B_tensor *= mask
    return B_tensor

def zero_col_bnd(B_tensor: torch.Tensor, col_bnds: torch.Tensor) -> torch.Tensor:
    b, m, n = B_tensor.shape
    device = B_tensor.device
    
    col_inds = torch.arange(n, device=device).reshape(1, n)  # Shape: (1, n)
    col_bnds = col_bnds.reshape(b, 1)                       # Shape: (b, 1)
    
    mask = col_inds < col_bnds                              # Broadcasting (b, 1, n)
    B_tensor *= mask.unsqueeze(1)                           # Shape: (b, m, n)
    return B_tensor

################################################################################################

def _square_dist_helper_exact_impl(A:torch.Tensor,B:torch.Tensor)->torch.Tensor:

    A_exp = A[...,None,:]
    B_exp = B[...,None,:,:]

    diff         = A_exp - B_exp
    squared_diff = torch.square(diff,out=diff)
    result       = torch.sum(squared_diff,dim=-1)
    return result

square_dist_helper_exact = torch.jit.script(_square_dist_helper_exact_impl)


def _set_self_interactions_to_value_impl(
    ZZ:torch.Tensor,self_dist_bool:bool,value:float=0.0
)->torch.Tensor:

    if (not self_dist_bool):
        return ZZ

    ZZ = ZZ.contiguous()
    ones_mask = torch.ones(ZZ.shape[-2],ZZ.shape[-1], dtype=torch.bool, device=ZZ.device)
    dia_mask  = torch.eye(ZZ.shape[-2],ZZ.shape[-1], dtype=torch.bool, device=ZZ.device)
    mask      = torch.logical_xor(ones_mask,dia_mask)

    if (ZZ.ndim == 3):
        mask  = mask.unsqueeze(0)

    ZZ.masked_fill_(torch.logical_not(mask,out=mask),value)
    return ZZ

set_self_interactions_to_value = torch.jit.script(_set_self_interactions_to_value_impl)


def _cdist_impl(XX:torch.Tensor,YY:torch.Tensor,self_dist_bool:bool)->torch.Tensor:
    ZZ = _square_dist_helper_exact_impl(XX,YY)
    if (XX.dtype == torch.float64):
        ZZ.clamp_(min=1e-15)
    else:
        ZZ.clamp_(min=1e-8)
    ZZ = torch.sqrt(ZZ)
    return ZZ


cdist = torch.jit.script(_cdist_impl)

################################################################################################

def lazy_cdist(X, Y):
    if not PYKEOPS_AVAILABLE:
        raise RuntimeError("PyKeOps is not available.")

    # Flatten batch dimension into the lazy tensor's symbolic operations
    X_i = LazyTensor(X[..., None, :])
    Y_j = LazyTensor(Y[..., None, :, :])

    # Compute pairwise squared distances
    D_ij = ((X_i - Y_j) ** 2).sum(-1) 

    # Compute Euclidean distances and return
    return D_ij.sqrt()

def lazy_id(m,n,dtype,device):
    if not PYKEOPS_AVAILABLE:
        raise RuntimeError("PyKeOps is not available.")

    i = torch.arange(m,device=device).to(dtype)
    j = torch.arange(n,device=device).to(dtype)

    i = LazyTensor(i[:,None,None])
    j = LazyTensor(j[None,:,None])

    return (0.5-(i-j)**2).step()

def lazy_mask(m,n,dtype,device):
    if not PYKEOPS_AVAILABLE:
        raise RuntimeError("PyKeOps is not available.")
    i = torch.ones(m,device=device,dtype=dtype,requires_grad=False)
    j = torch.zeros(n,device=device,dtype=dtype,requires_grad=False)

    i = LazyTensor(i[:,None,None])
    j = LazyTensor(j[None,:,None])

    return (i-j) - lazy_id(m,n,dtype,device)

def laplace_2d_lazy(q,XX,YY,self_dist_bool):
    if not PYKEOPS_AVAILABLE:
        return laplace_2d(q, XX, YY, 0.0, self_dist_bool)
    if (not self_dist_bool):
        ZZ = lazy_cdist(XX,YY)
        res = ZZ.log()
        return res @ q
    else:
        I  = lazy_id(XX.shape[1],YY.shape[1],\
            dtype=XX.dtype,device=XX.device)
        ZZ = lazy_cdist(XX,YY) + I
        res = ZZ.log()
        return res @ q

def laplace_2d(q,XX:torch.Tensor,YY:torch.Tensor,param:float,\
    self_dist_bool:bool)->torch.Tensor:
    
    assert param == 0
    assert XX.shape[-1] == 2
    assert YY.shape[-1] == 2

    ZZ = cdist(XX,YY,self_dist_bool)
    result = torch.log(ZZ,out=ZZ)
    result = set_self_interactions_to_value(result,self_dist_bool)
    return result @ q

def laplace_3d_lazy(q,XX,YY,self_dist_bool):
    if not PYKEOPS_AVAILABLE:
        return laplace_3d(q, XX, YY, 0.0, self_dist_bool)

    if (not self_dist_bool):
        ZZ = lazy_cdist(XX,YY)
        res = 1 / ZZ
        return res @ q
    else:

        M   = lazy_mask(XX.shape[1],YY.shape[1],\
            dtype=XX.dtype,device=XX.device)
        ZZ  = lazy_cdist(XX,YY) + 1e-15
        res = 1 / ZZ
        res = res * M
        return res @ q

def _laplace_3d_apply_impl(
    q:torch.Tensor,XX:torch.Tensor,YY:torch.Tensor,self_dist_bool:bool
)->torch.Tensor:
    ZZ     = _cdist_impl(XX,YY,self_dist_bool)
    result = torch.reciprocal(ZZ,out=ZZ)
    result = _set_self_interactions_to_value_impl(result,self_dist_bool)
    return torch.sum(result @ q, dim=-1).unsqueeze(-1)


laplace_3d_apply = torch.jit.script(_laplace_3d_apply_impl)
     

def laplace_3d(q,XX:torch.Tensor,YY:torch.Tensor,param:float,\
    self_dist_bool:bool)->torch.Tensor:
    
    assert param == 0
    assert XX.shape[-1] == 3
    assert YY.shape[-1] == 3
    assert q.shape[-1]  == 1

    return laplace_3d_apply(q,XX,YY,self_dist_bool)


def laplace_3d_double_layer_lazy(
    q,
    XX,
    YY,
    self_dist_bool,
    src_geom=None,
):
    if not PYKEOPS_AVAILABLE:
        return laplace_3d_double_layer(q, XX, YY, 0.0, self_dist_bool, src_geom=src_geom)
    if src_geom is None:
        raise ValueError("src_geom is required for the Laplace double-layer kernel")

    X_i = LazyTensor(XX[..., :, None, :])
    Y_j = LazyTensor(YY[..., None, :, :])
    G_j = LazyTensor(src_geom[..., None, :, :])

    diff = X_i - Y_j
    numer = (diff * G_j).sum(-1)
    dist = lazy_cdist(XX, YY)

    if self_dist_bool:
        I = lazy_id(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        M = lazy_mask(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        denom = (dist + I) ** 3
        result = (-(1.0 / (4.0 * torch.pi)) * numer / denom) * M
    else:
        result = -(1.0 / (4.0 * torch.pi)) * numer / (dist ** 3)

    return result @ q

def laplace_3d_double_layer(
    q,
    XX: torch.Tensor,
    YY: torch.Tensor,
    param: float,
    self_dist_bool: bool,
    src_geom: torch.Tensor = None,
) -> torch.Tensor:
    assert param == 0
    assert XX.shape[-1] == 3
    assert YY.shape[-1] == 3
    assert q.shape[-1] == 1
    if src_geom is None:
        raise ValueError("src_geom is required for the Laplace double-layer kernel")

    XX_exp = XX[..., :, None, :]
    YY_exp = YY[..., None, :, :]
    diff = XX_exp - YY_exp
    dist2 = torch.sum(diff * diff, dim=-1)
    min_dist2 = 1e-30 if XX.dtype == torch.float64 else 1e-12
    dist2 = torch.clamp(dist2, min=min_dist2)
    dist = torch.sqrt(dist2)
    rotn = torch.sum(diff * src_geom[..., None, :, :], dim=-1)
    result = -(1.0 / (4.0 * torch.pi)) * rotn / (dist2 * dist)
    result = set_self_interactions_to_value(result, self_dist_bool)
    return result @ q


def laplace_3d_adjoint_double_layer_lazy(
    q,
    XX,
    YY,
    self_dist_bool,
    trg_geom=None,
):
    if not PYKEOPS_AVAILABLE:
        return laplace_3d_adjoint_double_layer(q, XX, YY, 0.0, self_dist_bool, trg_geom=trg_geom)
    if trg_geom is None:
        raise ValueError("trg_geom is required for the Laplace adjoint double-layer kernel")

    X_i = LazyTensor(XX[..., :, None, :])
    Y_j = LazyTensor(YY[..., None, :, :])
    G_i = LazyTensor(trg_geom[..., :, None, :])

    diff = X_i - Y_j
    numer = (diff * G_i).sum(-1)
    dist = lazy_cdist(XX, YY)

    if self_dist_bool:
        I = lazy_id(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        M = lazy_mask(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        denom = (dist + I) ** 3
        result = ((1.0 / (4.0 * torch.pi)) * numer / denom) * M
    else:
        result = (1.0 / (4.0 * torch.pi)) * numer / (dist ** 3)

    return result @ q

def laplace_3d_adjoint_double_layer(
    q,
    XX: torch.Tensor,
    YY: torch.Tensor,
    param: float,
    self_dist_bool: bool,
    trg_geom: torch.Tensor = None,
) -> torch.Tensor:
    assert param == 0
    assert XX.shape[-1] == 3
    assert YY.shape[-1] == 3
    assert q.shape[-1] == 1
    if trg_geom is None:
        raise ValueError("trg_geom is required for the Laplace adjoint double-layer kernel")

    XX_exp = XX[..., :, None, :]
    YY_exp = YY[..., None, :, :]
    diff = XX_exp - YY_exp
    dist2 = torch.sum(diff * diff, dim=-1)
    min_dist2 = 1e-30 if XX.dtype == torch.float64 else 1e-12
    dist2 = torch.clamp(dist2, min=min_dist2)
    dist = torch.sqrt(dist2)
    rotn = torch.sum(diff * trg_geom[..., :, None, :], dim=-1)
    result = (1.0 / (4.0 * torch.pi)) * rotn / (dist2 * dist)
    result = set_self_interactions_to_value(result, self_dist_bool)
    return result @ q


def helmholtz_3d_double_layer_lazy(
    q,
    XX,
    YY,
    kappa,
    self_dist_bool,
    src_geom=None,
):
    if not PYKEOPS_AVAILABLE:
        return helmholtz_3d_double_layer(q, XX, YY, kappa, self_dist_bool, src_geom=src_geom)
    if src_geom is None:
        raise ValueError("src_geom is required for the Helmholtz double-layer kernel")

    q_real = torch.real(q).contiguous()
    q_imag = torch.imag(q).contiguous()

    X_i = LazyTensor(XX[..., :, None, :])
    Y_j = LazyTensor(YY[..., None, :, :])
    G_j = LazyTensor(src_geom[..., None, :, :])

    diff = X_i - Y_j
    numer = (diff * G_j).sum(-1)
    dist = lazy_cdist(XX, YY)

    if self_dist_bool:
        I = lazy_id(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        M = lazy_mask(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        dist_safe = dist + I
    else:
        M = None
        dist_safe = dist

    kdist = kappa * dist_safe
    inv_r3 = dist_safe ** (-3)
    real_factor = -(kdist.cos() + kdist * kdist.sin()) * inv_r3
    imag_factor = (kdist * kdist.cos() - kdist.sin()) * inv_r3
    kernel_real = (1.0 / (4.0 * torch.pi)) * numer * real_factor
    kernel_imag = (1.0 / (4.0 * torch.pi)) * numer * imag_factor

    if self_dist_bool:
        kernel_real = kernel_real * M
        kernel_imag = kernel_imag * M

    res_real = kernel_real @ q_real
    res_real -= kernel_imag @ q_imag
    res_imag = kernel_real @ q_imag
    res_imag += kernel_imag @ q_real
    return torch.complex(res_real, res_imag)


def helmholtz_3d_double_layer(
    q,
    XX: torch.Tensor,
    YY: torch.Tensor,
    kappa: float,
    self_dist_bool: bool,
    src_geom: torch.Tensor = None,
) -> torch.Tensor:
    assert kappa > 0
    assert XX.shape[-1] == 3
    assert YY.shape[-1] == 3
    assert q.shape[-1] == 1
    if src_geom is None:
        raise ValueError("src_geom is required for the Helmholtz double-layer kernel")

    dist = cdist(XX, YY, self_dist_bool)
    c_dtype = torch.cdouble if XX.dtype == torch.float64 else torch.cfloat
    diff = XX[..., :, None, :] - YY[..., None, :, :]
    rotn = torch.sum(diff * src_geom[..., None, :, :], dim=-1)

    dist_complex = dist.to(c_dtype)
    kdist = kappa * dist_complex
    factor = torch.exp(1j * kdist) * (1j * kdist - 1.0)
    factor = factor / (dist_complex * dist_complex * dist_complex)
    result = (1.0 / (4.0 * torch.pi)) * rotn.to(c_dtype) * factor
    result = set_self_interactions_to_value(result, self_dist_bool)
    return result @ q.to(c_dtype)


def helmholtz_3d_adjoint_double_layer_lazy(
    q,
    XX,
    YY,
    kappa,
    self_dist_bool,
    trg_geom=None,
):
    if not PYKEOPS_AVAILABLE:
        return helmholtz_3d_adjoint_double_layer(q, XX, YY, kappa, self_dist_bool, trg_geom=trg_geom)
    if trg_geom is None:
        raise ValueError("trg_geom is required for the Helmholtz adjoint double-layer kernel")

    q_real = torch.real(q).contiguous()
    q_imag = torch.imag(q).contiguous()

    X_i = LazyTensor(XX[..., :, None, :])
    Y_j = LazyTensor(YY[..., None, :, :])
    G_i = LazyTensor(trg_geom[..., :, None, :])

    diff = X_i - Y_j
    numer = (diff * G_i).sum(-1)
    dist = lazy_cdist(XX, YY)

    if self_dist_bool:
        I = lazy_id(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        M = lazy_mask(XX.shape[1], YY.shape[1], dtype=XX.dtype, device=XX.device)
        dist_safe = dist + I
    else:
        M = None
        dist_safe = dist

    kdist = kappa * dist_safe
    inv_r3 = dist_safe ** (-3)
    real_factor = -(kdist.cos() + kdist * kdist.sin()) * inv_r3
    imag_factor = (kdist * kdist.cos() - kdist.sin()) * inv_r3
    kernel_real = -(1.0 / (4.0 * torch.pi)) * numer * real_factor
    kernel_imag = -(1.0 / (4.0 * torch.pi)) * numer * imag_factor

    if self_dist_bool:
        kernel_real = kernel_real * M
        kernel_imag = kernel_imag * M

    res_real = kernel_real @ q_real
    res_real -= kernel_imag @ q_imag
    res_imag = kernel_real @ q_imag
    res_imag += kernel_imag @ q_real
    return torch.complex(res_real, res_imag)


def helmholtz_3d_adjoint_double_layer(
    q,
    XX: torch.Tensor,
    YY: torch.Tensor,
    kappa: float,
    self_dist_bool: bool,
    trg_geom: torch.Tensor = None,
) -> torch.Tensor:
    assert kappa > 0
    assert XX.shape[-1] == 3
    assert YY.shape[-1] == 3
    assert q.shape[-1] == 1
    if trg_geom is None:
        raise ValueError("trg_geom is required for the Helmholtz adjoint double-layer kernel")

    dist = cdist(XX, YY, self_dist_bool)
    c_dtype = torch.cdouble if XX.dtype == torch.float64 else torch.cfloat
    diff = XX[..., :, None, :] - YY[..., None, :, :]
    rotn = torch.sum(diff * trg_geom[..., :, None, :], dim=-1)

    dist_complex = dist.to(c_dtype)
    kdist = kappa * dist_complex
    factor = torch.exp(1j * kdist) * (1j * kdist - 1.0)
    factor = factor / (dist_complex * dist_complex * dist_complex)
    result = -(1.0 / (4.0 * torch.pi)) * rotn.to(c_dtype) * factor
    result = set_self_interactions_to_value(result, self_dist_bool)
    return result @ q.to(c_dtype)


################################################################################################

def j0_inplace(ZZ):
    return torch.special.bessel_j0(ZZ, out=ZZ)
 
def y0_inplace(ZZ):
    return torch.special.bessel_y0(ZZ, out=ZZ)

def helmholtz_2d(q,XX,YY,kappa,self_dist_bool):
    
    assert kappa > 0;
    assert q.shape[-1]  == 1
    assert XX.shape[-1] == 2
    assert YY.shape[-1] == 2

    device = XX.device
    ZZ     = cdist(XX,YY,self_dist_bool)
    dtype  = torch.complex128 if XX.dtype == torch.float64 else torch.complex64
    result = torch.zeros(ZZ.shape,dtype=dtype,device=device,requires_grad=False)
    
    ZZ *= kappa
    ZZ_copy = ZZ.clone()

    result += y0_inplace(ZZ)
    result *= 1j
    result += j0_inplace(ZZ_copy)

    result = set_self_interactions_to_value(result,self_dist_bool)
    return result @ q

def helmholtz_3d_lazy(q,XX,YY,kappa,self_dist_bool):
    if not PYKEOPS_AVAILABLE:
        return helmholtz_3d(q, XX, YY, kappa, self_dist_bool)
    q_real = torch.real(q).contiguous()
    q_imag = torch.imag(q).contiguous()
    if (not self_dist_bool):

        ZZ   = lazy_cdist(XX,YY)
        kZZ  = kappa*ZZ
        Dcos = kZZ.cos() / ZZ
        Dsin = kZZ.sin() / ZZ

        res_real = Dcos @ q_real
        res_real-= Dsin @ q_imag

        res_imag = Dcos @ q_imag
        res_imag+= Dsin @ q_real
        return torch.complex(res_real,res_imag)

    else:

        ZZ   = lazy_cdist(XX,YY)
        kZZ  = kappa * ZZ
        M    = lazy_mask(XX.shape[1],YY.shape[1],\
            dtype=XX.dtype,device=XX.device)
        Denom= 1 / (ZZ + 1e-15)
        Denom= Denom * M

        Dcos = kZZ.cos() * Denom
        Dsin = kZZ.sin() * Denom

        Dcos = Dcos * M
        Dsin = Dsin * M

        res_real = Dcos @ q_real
        res_real-= Dsin @ q_imag

        res_imag = Dcos @ q_imag
        res_imag+= Dsin @ q_real
        return torch.complex(res_real,res_imag)

def helmholtz_3d(q,XX:torch.Tensor,YY:torch.Tensor,kappa:float,\
    self_dist_bool:bool)->torch.Tensor:
    
    assert kappa > 0
    assert XX.shape[-1] == 3
    assert YY.shape[-1] == 3
    assert q.shape[-1]  == 1
    device = XX.device

    ZZ      = cdist(XX,YY,self_dist_bool)
    c_dtype = torch.cdouble if ZZ.dtype == torch.double else torch.cfloat

    ZZ_complex  = ZZ.to(c_dtype)
    ZZ_complex *= (+1j*kappa)
    
    result = torch.exp(ZZ_complex,out=ZZ_complex)
    result = torch.div(result,ZZ,out=result)

    result = set_self_interactions_to_value(result,self_dist_bool)
    return result @ q
