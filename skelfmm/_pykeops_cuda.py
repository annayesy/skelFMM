import os
import site
import sysconfig
from pathlib import Path


def _prepend_env_path(env, key, values):
    values = [str(Path(value)) for value in values if value]
    if not values:
        return
    existing = [entry for entry in env.get(key, "").split(os.pathsep) if entry]
    merged = []
    for entry in values + existing:
        if entry not in merged:
            merged.append(entry)
    env[key] = os.pathsep.join(merged)


def _candidate_cuda_roots(cuda_version, *, conda_prefix=None):
    roots = []
    env = os.environ
    for key in ("CUDA_PATH", "CUDA_HOME"):
        value = env.get(key)
        if value:
            roots.append(Path(value))

    if conda_prefix:
        roots.append(Path(conda_prefix))

    if cuda_version:
        major_minor = ".".join(str(cuda_version).split(".")[:2])
        roots.append(Path(f"/usr/local/cuda-{major_minor}"))
    roots.extend((Path("/usr/local/cuda"), Path("/opt/cuda")))

    unique = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _candidate_nvidia_site_roots(site_roots=None):
    roots = []
    if site_roots is None:
        site_roots = []
        try:
            site_roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            purelib = sysconfig.get_paths().get("purelib")
            if purelib:
                site_roots.append(purelib)
        except Exception:
            pass

    unique_site_roots = []
    for root in site_roots:
        path = Path(root)
        if path not in unique_site_roots:
            unique_site_roots.append(path)

    for site_root in unique_site_roots:
        nvidia_root = site_root / "nvidia"
        if nvidia_root.is_dir():
            roots.append(nvidia_root)
    return roots


def _discover_cuda_layout(cuda_version, *, conda_prefix=None, search_roots=None, nvidia_site_roots=None):
    include_dirs = []
    lib_dirs = []
    bin_dirs = []
    cuda_root = None

    if search_roots is None:
        search_roots = _candidate_cuda_roots(cuda_version, conda_prefix=conda_prefix)

    for root in search_roots:
        include_dir = root / "include"
        lib64_dir = root / "lib64"
        lib_dir = root / "lib"
        bin_dir = root / "bin"

        has_cuda_h = (include_dir / "cuda.h").is_file()
        has_nvrtc_h = (include_dir / "nvrtc.h").is_file()
        if has_cuda_h and has_nvrtc_h:
            cuda_root = root
            include_dirs.append(include_dir)
            if lib64_dir.is_dir():
                lib_dirs.append(lib64_dir)
            if lib_dir.is_dir():
                lib_dirs.append(lib_dir)
            if bin_dir.is_dir():
                bin_dirs.append(bin_dir)
            break

    for nvidia_root in _candidate_nvidia_site_roots(site_roots=nvidia_site_roots):
        runtime_include = nvidia_root / "cuda_runtime" / "include"
        nvrtc_include = nvidia_root / "cuda_nvrtc" / "include"
        runtime_lib = nvidia_root / "cuda_runtime" / "lib"
        nvrtc_lib = nvidia_root / "cuda_nvrtc" / "lib"

        if (runtime_include / "cuda.h").is_file():
            include_dirs.append(runtime_include)
        if (nvrtc_include / "nvrtc.h").is_file():
            include_dirs.append(nvrtc_include)
        if runtime_lib.is_dir():
            lib_dirs.append(runtime_lib)
        if nvrtc_lib.is_dir():
            lib_dirs.append(nvrtc_lib)

    # Preserve order while deduplicating.
    include_dirs = list(dict.fromkeys(include_dirs))
    lib_dirs = list(dict.fromkeys(lib_dirs))
    bin_dirs = list(dict.fromkeys(bin_dirs))
    return {
        "cuda_root": cuda_root,
        "include_dirs": include_dirs,
        "lib_dirs": lib_dirs,
        "bin_dirs": bin_dirs,
    }


def configure_pykeops_cuda_environment(env=None):
    """
    Populate CUDA-related environment variables before importing PyKeOps.

    This is a best-effort helper that uses the active Torch/CUDA setup to
    populate ``CUDA_PATH`` / ``CUDA_HOME`` when a full toolkit root is
    available, and otherwise extends library and include search paths so that
    PyKeOps can discover CUDA runtime and NVRTC components automatically.
    """
    if env is None:
        env = os.environ

    if env.get("SKELFMM_SKIP_PYKEOPS_CUDA_SETUP") == "1":
        return None

    try:
        import torch
    except Exception:
        return None

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return None

    layout = _discover_cuda_layout(cuda_version, conda_prefix=env.get("CONDA_PREFIX"))

    cuda_root = layout["cuda_root"]
    if cuda_root is not None:
        env.setdefault("CUDA_PATH", str(cuda_root))
        env.setdefault("CUDA_HOME", str(cuda_root))

    _prepend_env_path(env, "LD_LIBRARY_PATH", layout["lib_dirs"])
    _prepend_env_path(env, "LIBRARY_PATH", layout["lib_dirs"])
    _prepend_env_path(env, "DYLD_LIBRARY_PATH", layout["lib_dirs"])
    _prepend_env_path(env, "PATH", layout["bin_dirs"])
    _prepend_env_path(env, "CPATH", layout["include_dirs"])
    _prepend_env_path(env, "CPLUS_INCLUDE_PATH", layout["include_dirs"])

    return layout
