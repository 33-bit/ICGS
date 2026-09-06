"""Sequential native proposals: K is not the batch B or horizon P, and is not search."""
from dataclasses import dataclass
from typing import Any
import time
import numpy as np
from icgs.state.randomness import scoped_seed


@dataclass(frozen=True)
class Candidate:
    index: int
    trajectory: Any
    root_pose: np.ndarray
    context_id: str
    source_id: str
    seed: int | None
    seconds: float
    batch_size: int
    horizon: int


def propose_candidates(proposer, observation, context, *, count, seeds=None):
    if isinstance(count,bool) or not isinstance(count,int) or count<1:raise ValueError('candidate count must be positive integer')
    if seeds is not None and len(seeds)!=count:raise ValueError('one seed per candidate required')
    proposer.validate_context(context)
    results=[]
    for index in range(count):
        seed=None if seeds is None else seeds[index]
        start=time.perf_counter()
        with scoped_seed(seed,device=proposer.runtime.device):
            trajectory=proposer.predict(observation,context)
        # CUDA synchronization ensures accounting measures completed prediction.
        import torch
        if torch.device(proposer.runtime.device).type=='cuda':torch.cuda.synchronize()
        results.append(Candidate(index,trajectory,np.array(observation.T_w_e,copy=True),
                                 getattr(context,'source_id','unrecorded'),getattr(proposer,'artifact_sha256','unrecorded'),
                                 seed,time.perf_counter()-start,trajectory.transforms.shape[0],trajectory.transforms.shape[1]))
    return results
