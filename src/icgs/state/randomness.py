"""Scoped RNG for serial inference; not safe for concurrent global RNG consumers."""
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def scoped_seed(seed, *, device='cpu'):
    if seed is None:
        yield
        return
    python_state=random.getstate(); numpy_state=np.random.get_state()
    cuda=torch.device(device).type=='cuda'
    devices=list(range(torch.cuda.device_count())) if cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
            if cuda: torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
