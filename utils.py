import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int] = 42,
                    deterministic: bool = True,
                    warn: bool = True) -> int:
    """
    Seed Python, NumPy, PyTorch (CPU & CUDA) and set relevant
    environment variables so results become *as* reproducible
    as PyTorch allows.

    Parameters
    ----------
    seed          : int or None
        Seed to use.  If None, a time-based seed is generated and returned.
    deterministic : bool
        • True  → use deterministic algorithms whenever possible  
        • False → allow the fastest (non-deterministic) algorithms
    warn          : bool
        Print a short note when `deterministic=False`.

    Returns
    -------
    int
        The seed actually used (handy if you passed None).

    Notes
    -----
    • Full determinism can reduce speed and may **limit which operations are
      allowed**; PyTorch will raise an error if you call an op that does not
      have a deterministic implementation when `deterministic=True`.
    • For CUDA ≥ 10.2 you *also* need to set the environment variable
      `CUBLAS_WORKSPACE_CONFIG` **before** the first CUDA context is created
      to guarantee bit-wise repeatability of some GEMM kernels.
    """
    if seed is None:
        # nanosecond-precision clock gives a reasonably unique seed
        seed = int(datetime.utcnow().timestamp() * 1e9) & 0xFFFFFFFF

    # ------------------------------------------------------------------
    # Basic RNGs
    # ------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)      # affects hashing of str() etc.
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # PyTorch RNGs
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)              # if you use multi-GPU / DDP

    # ------------------------------------------------------------------
    # PyTorch deterministic / benchmark flags
    # ------------------------------------------------------------------
    if deterministic:
        # (1) Force cuDNN and other backend libraries into deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False   # True → faster but random

        # (2) Tell PyTorch to error out if a deterministic implementation
        #     of an operation is not available.
        torch.use_deterministic_algorithms(True)

        # (3) Extra flag for cuBLAS (needed for some matrix-mult kernels)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        if warn:
            print("[seed_everything] Non-deterministic algorithms are allowed "
                  "(deterministic=False).  Results may vary from run to run.")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
        torch.use_deterministic_algorithms(False)

    return seed