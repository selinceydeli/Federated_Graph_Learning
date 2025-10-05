from __future__ import annotations
import os
import random
import hashlib
import numpy as np
import torch
from contextlib import contextmanager

def set_seed(
    seed: int,
    *,
    deterministic_torch: bool = True,
    set_python_hashseed: bool = False,
) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    if set_python_hashseed:
        os.environ["PYTHONHASHSEED"] = str(seed) 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # To set strict determinism for all operations, uncomment:
        # torch.use_deterministic_algorithms(True)


def derive_seed(base: int, name: str) -> int:
    """
    Derive a stable per-split/per-run seed from a base seed and a label.
    """
    h = hashlib.blake2b(f"{base}:{name}".encode(), digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)


def seed_worker(worker_id: int) -> None:
    """
    For use with PyTorch DataLoader workers:
        DataLoader(..., worker_init_fn=seed_worker, generator=gen)
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@contextmanager
def torch_determinism():
    """
    Context manager to temporarily force deterministic cuDNN settings.
    """
    prev_det = torch.backends.cudnn.deterministic
    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_det
        torch.backends.cudnn.benchmark = prev_bench
