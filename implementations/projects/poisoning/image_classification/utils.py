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

