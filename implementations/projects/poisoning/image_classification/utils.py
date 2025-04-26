from __future__ import annotations

import os
import random
import numpy as np
import torch
from functools import partialmethod
from tqdm.auto import tqdm, trange

def use_tqdm(enable: bool, ascii=False):
    """
    Call `use_tqdm(True)` for interactive output.
    
    Great for tracking training progress when plotting performance curves.

    Jupyter notebook issues: 
    - Might slow down the notebook
    - Does not display well after closing the notebook
    """
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not enable, ascii=ascii)

def seed_all_generators(seed):
    """
    Seed all the random number generators and use deterministic algorithms
    for reproducibility.

    Bear in mind that using deterministic algorithms might degrade performance.
    """
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def enough_memory_for_cdist_mm(matrix: torch.Tensor) -> bool:
    size = matrix.nbytes
    additional = 3 * size
    if additional < 64 * 1024 * 1024:
        return True

    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0]
    else:
        import psutil
        mem = psutil.virtual_memory()
        free_memory = mem.available

    return free_memory > additional

def mem_aware_pdist(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise distance matrix, but is compatible with half-precision
    and chooses the best computation method under memory constraints.

    Args:
        matrix (Tensor): a matrix containing row vectors.

    Returns:
        pairwise_distances (Tensor): the pairwise distance matrix.
    """
    if enough_memory_for_cdist_mm(matrix):
        compute_mode = "use_mm_for_euclid_dist_if_necessary"
    else:
        compute_mode = "donot_use_mm_for_euclid_dist"
    dist = torch.cdist(matrix, matrix, compute_mode=compute_mode)
    dist.fill_diagonal_(0.0)
    return dist