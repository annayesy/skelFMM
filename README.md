# skelFMM: A Simplified Kernel-Independent Fast Multipole Method (FMM)

[![DOI](https://zenodo.org/badge/905164703.svg)](https://doi.org/10.5281/zenodo.14613532)
[![License](https://img.shields.io/github/license/annayesy/skelFMM)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/annayesy/skelFMM)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/annayesy/skelFMM)
[![Latest commit](https://img.shields.io/github/last-commit/annayesy/skelFMM)](https://github.com/annayesy/skelFMM/commits/main)

## Overview

`skelFMM` is a research implementation of a novel kernel-independent fast multipole method (FMM) designed for efficiently evaluating discrete convolution kernels with given source distributions. This method introduces a simplified approach that eliminates the need for explicit interaction lists by leveraging near-neighbor computations at each level of an adaptive tree structure. The algorithm is well-suited for parallelization on modern hardware and supports a wide range of kernels.

Unlike traditional FMM implementations, `skelFMM` simplifies data structures by operating exclusively on the near-neighbor list (limited to a maximum size of 27 in 3D) rather than the interaction list (which can reach up to 189 in 3D). This makes the implementation lightweight and efficient while retaining full compatibility with adaptive quad-tree and octree structures. The method also introduces novel translation operators to handle adaptive point distributions effectively.

---

## Key Features

- **Simplified Data Structures**: Avoids the need for complex interaction lists, reducing implementation complexity.
- **Kernel Independence**: Requires only kernel evaluations, making it applicable to a wide variety of convolution kernels.
- **Parallel Efficiency**: Designed for modern hardware with GPU-accelerated batched linear algebra operations.
- **Adaptive Tree Compatibility**: Supports both uniform and non-uniform point distributions in 2D and 3D environments.
- **Precomputation Optimization**: Constructs tailored skeleton representations during a precomputation stage for efficient runtime performance.

<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">

  <figure style="margin: 0 20px; text-align: center;">
    <img src="https://raw.githubusercontent.com/annayesy/skelFMM/main/figures/interaction_list_square.png" alt="Interaction List Square" style="height: 200px; object-fit: contain;">
  </figure>

  <figure style="margin: 0 20px; text-align: center;">
    <img src="https://raw.githubusercontent.com/annayesy/skelFMM/main/figures/interaction_list_curvy_annulus.png" alt="Interaction List Curvy Annulus" style="height: 200px; object-fit: contain;">
  </figure>
</div>

The figures show the interaction list of size at most 27 in 2D, as well as additional lists, which are typically part of traditional FMM implementations.
The work reorganizes the computations involved in the kernel-independent FMM to traverse the near-neighbor list at every level of the tree, which retaining full compatatibility with adaptive tree structures.

---

## Citation

If you use `skelFMM` in your research, please cite the accompanying paper:

```
@article{yesypenko2024simplified,
  title={A simplified fast multipole method based on strong recursive skeletonization},
  author={Yesypenko, Anna and Chen, Chao and Martinsson, Per-Gunnar},
  journal={Journal of Computational Physics},
  pages={113707},
  year={2024},
  publisher={Elsevier}
}
```

