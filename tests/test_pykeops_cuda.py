from pathlib import Path

from skelfmm._pykeops_cuda import _discover_cuda_layout, _prepend_env_path


def test_discover_cuda_layout_prefers_full_toolkit_root(tmp_path):
    cuda_root = tmp_path / "cuda-12.1"
    (cuda_root / "include").mkdir(parents=True)
    (cuda_root / "lib64").mkdir()
    (cuda_root / "bin").mkdir()
    (cuda_root / "include" / "cuda.h").write_text("", encoding="utf-8")
    (cuda_root / "include" / "nvrtc.h").write_text("", encoding="utf-8")

    layout = _discover_cuda_layout(
        "12.1",
        search_roots=[cuda_root],
        nvidia_site_roots=[],
    )

    assert layout["cuda_root"] == cuda_root
    assert layout["include_dirs"] == [cuda_root / "include"]
    assert layout["lib_dirs"] == [cuda_root / "lib64"]
    assert layout["bin_dirs"] == [cuda_root / "bin"]


def test_discover_cuda_layout_collects_split_nvidia_site_packages(tmp_path):
    site_root = tmp_path / "site-packages"
    runtime_include = site_root / "nvidia" / "cuda_runtime" / "include"
    nvrtc_include = site_root / "nvidia" / "cuda_nvrtc" / "include"
    runtime_lib = site_root / "nvidia" / "cuda_runtime" / "lib"
    nvrtc_lib = site_root / "nvidia" / "cuda_nvrtc" / "lib"

    runtime_include.mkdir(parents=True)
    nvrtc_include.mkdir(parents=True)
    runtime_lib.mkdir(parents=True)
    nvrtc_lib.mkdir(parents=True)
    (runtime_include / "cuda.h").write_text("", encoding="utf-8")
    (nvrtc_include / "nvrtc.h").write_text("", encoding="utf-8")

    layout = _discover_cuda_layout(
        "12.1",
        search_roots=[],
        nvidia_site_roots=[site_root],
    )

    assert layout["cuda_root"] is None
    assert runtime_include in layout["include_dirs"]
    assert nvrtc_include in layout["include_dirs"]
    assert runtime_lib in layout["lib_dirs"]
    assert nvrtc_lib in layout["lib_dirs"]


def test_prepend_env_path_deduplicates_entries():
    env = {"LD_LIBRARY_PATH": "/b:/c"}
    _prepend_env_path(env, "LD_LIBRARY_PATH", [Path("/a"), Path("/b")])
    assert env["LD_LIBRARY_PATH"] == "/a:/b:/c"
