"""Acceptance job: strict published weights, real GPU inference, fidelity and standalone.

Run reference/cross-process comparisons before native-only standalone probe. Synthetic
fixture establishes execution/fidelity only, never benchmark success.
"""
import copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import instant_policy
from utils import transform_pcd,subsample_pcd
from icgs.artifacts.published import load_published_policy
from icgs.data.schemas.inference import load_input
from icgs.algorithms.planning.candidates import propose_candidates
from icgs.execution.commands import absolute_prefix
from icgs.state.randomness import scoped_seed

torch.set_num_threads(2)
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--fixture',required=True)
parser.add_argument('--output-directory',required=True)
args=parser.parse_args()
root=Path(args.output_directory);root.mkdir(parents=True,exist_ok=True)
checkpoint=args.checkpoint
ref=instant_policy.GraphDiffusion.load_from_checkpoint(checkpoint,device='cuda',strict=True,map_location='cuda')
ref.set_num_demos(2);ref.set_num_diffusion_steps(4);ref.eval()
native=load_published_policy(checkpoint)
observation,demos=load_input(args.fixture)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def reference(s):
 seed(s)
 processed=[instant_policy.sample_to_cond_demo(d,10) for d in copy.deepcopy(demos)]
 sample={'demos':processed,'live':{'obs':[transform_pcd(subsample_pcd(observation.points),np.linalg.inv(observation.T_w_e))],
                                 'grips':[observation.grip],'T_w_es':[observation.T_w_e]}}
 with torch.no_grad():a,g=ref.predict_actions(sample)
 return (a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else np.asarray(a),
         g.detach().cpu().numpy() if isinstance(g,torch.Tensor) else np.asarray(g))
def native_cold(s):
 with scoped_seed(s,device='cuda'):
  c=native.prepare_context(copy.deepcopy(demos));r=native.predict(observation,c)
 return r,c
def errors(a,b,ga,gb):
 a=a.reshape(-1,4,4);b=b.reshape(-1,4,4)
 rot=Rotation.from_matrix(a[:,:3,:3]).inv()*Rotation.from_matrix(b[:,:3,:3])
 return dict(matrix_max=float(abs(a-b).max()),matrix_mean=float(abs(a-b).mean()),
             translation_max=float(abs(a[:,:3,3]-b[:,:3,3]).max()),rotation_rad_max=float(rot.magnitude().max()),
             grip_equal=bool(np.array_equal(ga.reshape(-1),gb.reshape(-1))))
repeat=[];pairs={}
for s in [17,29,41]:
 a,g=reference(s);b,h=reference(s);repeat.append(errors(a,b,g,h));pairs[s]=(a,g)
# Tolerance set from reference repeatability only before looking at native errors.
matrix_tol=max(5e-6,20*max(x['matrix_max'] for x in repeat))
translation_tol=max(1e-6,20*max(x['translation_max'] for x in repeat))
rotation_tol=max(1e-5,20*max(x['rotation_rad_max'] for x in repeat))
records=[]
for s in [17,29,41]:
 result,context=native_cold(s);a,g=pairs[s]
 diff=errors(a,result.transforms.cpu().numpy(),g,result.grips.cpu().numpy())
 assert diff['matrix_max']<=matrix_tol and diff['translation_max']<=translation_tol and diff['rotation_rad_max']<=rotation_tol and diff['grip_equal'],diff
 assert not result.transforms.requires_grad and not result.grips.requires_grad
 records.append(dict(seed=s,**diff))
 # Warm and reset paths: fixed prepared context, identical seed and no hidden persistent features.
 with scoped_seed(s,device='cuda'):warm=native.predict(observation,context)
 native.reset_context(context)
 with scoped_seed(s,device='cuda'):reset=native.predict(observation,context)
 assert torch.allclose(warm.transforms,reset.transforms,atol=matrix_tol,rtol=0)
 assert torch.equal(warm.grips,reset.grips)
 before=torch.cuda.get_rng_state().clone()
 candidates=propose_candidates(native,observation,context,count=3,seeds=[s,s+1,s+2])
 assert torch.equal(before,torch.cuda.get_rng_state())
 assert len(candidates)==3 and [x.index for x in candidates]==[0,1,2]
 assert torch.allclose(candidates[0].trajectory.transforms,reset.transforms,atol=matrix_tol,rtol=0)
 prefix=absolute_prefix(candidates[0].trajectory,observation.T_w_e,3)
 np.testing.assert_allclose(prefix.targets,observation.T_w_e@reset.transforms.cpu().numpy()[:,:3],atol=matrix_tol,rtol=0)
report=dict(status='PASS',checkpoint_sha256=native.artifact_sha256,
            C1='PASS',C2='PASS',C3='PASS',C4='PASS',C5='requires separate native-only CLI subprocess',
            reference_repeatability=repeat,tolerances=dict(matrix=matrix_tol,translation=translation_tol,rotation_rad=rotation_tol),
            comparisons=records,candidate_tests='K=1 parity and K=3 count/index/RNG/prefix PASS',
            context='cold/warm/reset PASS; published profile does not persist scene embeddings',
            input='geometric synthetic two-demo nontrivial-pose fixture; no task success claim')
Path(root/'acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
