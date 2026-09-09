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


def propose_candidates(proposer, observation, context, *, count, seeds=None, recorder=None):
    if isinstance(count,bool) or not isinstance(count,int) or count<1:raise ValueError('candidate count must be positive integer')
    if seeds is not None and len(seeds)!=count:raise ValueError('one seed per candidate required')
    if recorder is None:
        from icgs.observability.recorder import NoopRecorder
        recorder = NoopRecorder()
    proposer.validate_context(context)
    results=[]
    for index in range(count):
        seed=None if seeds is None else seeds[index]
        start=time.perf_counter()
        source_id = getattr(context,'source_id','unrecorded')
        artifact_id = getattr(proposer,'artifact_sha256','unrecorded')
        bound = recorder.bind(candidate_id=str(index), origin="real")
        with bound.span("policy.propose", component="proposal",
                        fields={"index": index, "seed": seed, "source_id": source_id,
                                "artifact_id": artifact_id}):
            with scoped_seed(seed,device=proposer.runtime.device):
                trajectory=proposer.predict(observation,context)
            # CUDA synchronization ensures accounting measures completed prediction.
            import torch
            if torch.device(proposer.runtime.device).type=='cuda':torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        result = Candidate(index,trajectory,np.array(observation.T_w_e,copy=True),
                           source_id,artifact_id,seed,seconds,
                           trajectory.transforms.shape[0],trajectory.transforms.shape[1])
        bound.event("candidate.created", component="proposal",
                    fields={"index": index, "seed": seed, "source_id": source_id,
                            "artifact_id": artifact_id, "seconds": seconds,
                            "batch_size": result.batch_size, "horizon": result.horizon})
        results.append(result)
    return results
