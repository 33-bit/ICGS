"""Public inference facade with per-policy, episode-local demonstration state."""
import torch
from icgs.contracts.records import Observation, ActionTrajectory
from icgs.state.context_cache import PreparedContext


class InstantPolicy:
    def __init__(self, network, sampler, objective, graph_config, runtime):
        self.network, self.sampler, self.objective = network, sampler, objective
        self.graph_config, self.runtime = graph_config, runtime
        self.context_owner = object()

    def eval(self):
        self.network.eval()
        return self

    def train(self, mode=True):
        self.network.train(mode)
        return self

    def prepare_context(self, demos, *, prepared=False):
        import hashlib
        import numpy as np
        from types import MappingProxyType
        if len(demos) != self.graph_config.num_demos:
            raise ValueError('demonstration count does not match policy graph')
        if not prepared:
            from icgs.data.preprocessing.native import sample_to_cond_demo
            demos = [sample_to_cond_demo(demo, self.graph_config.traj_horizon) for demo in demos]
        for demo in demos:
            if any(len(demo.get(key,[]))!=self.graph_config.traj_horizon for key in ('obs','grips','T_w_es')):
                raise ValueError('demonstration waypoint count does not match policy graph')
        # Do not run encoder early: original evaluation encodes on first predict per rollout.
        # Own content; immutable backing bytes prevent callers re-enabling writes.
        frozen=[]
        for demo in demos:
            content={}
            for key in ('obs','grips','T_w_es'):
                values=[]
                for value in demo[key]:
                    array=np.ascontiguousarray(value)
                    if array.dtype.hasobject:raise ValueError('demonstration content must be numeric')
                    values.append(np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape))
                content[key]=tuple(values)
            frozen.append(MappingProxyType(content))
        demos=tuple(frozen)
        digest=hashlib.sha256()
        for demo in demos:
            for key in ('obs','grips','T_w_es'):
                for value in demo[key]:
                    array=np.ascontiguousarray(value)
                    digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
        return PreparedContext(demos, self.context_owner, source_id=digest.hexdigest())

    def validate_context(self, context):
        if context.owner is not self.context_owner:
            raise ValueError('context belongs to another policy; prepare it again')
        if len(context.demos)!=self.graph_config.num_demos:
            raise ValueError('context demo count changed; prepare it again')
        if any(len(d.get(key,[]))!=self.graph_config.traj_horizon for d in context.demos for key in ('obs','grips','T_w_es')):
            raise ValueError('context waypoint count changed; prepare it again')

    def reset_context(self, context):
        self.validate_context(context)
        context.embeddings = context.positions = None

    def predict_batch(self, data):
        """Model-ready batch API; caller data is not mutated by sampling."""
        data = data.clone().to(self.runtime.device)
        if (self.network.batch_size != data.actions.shape[0]
                or self.network.num_demos != self.graph_config.num_demos):
            self.network.reinit_graphs(data.actions.shape[0], num_demos=self.graph_config.num_demos)
        with torch.no_grad():
            device_type = torch.device(self.runtime.device).type
            with torch.autocast(dtype=torch.float32, device_type=device_type, enabled=device_type == 'cuda'):
                return self.sampler.sample(self.network, data)

    @torch.no_grad()
    def predict(self, observation: Observation, context: PreparedContext) -> ActionTrajectory:
        import numpy as np
        from icgs.data.preprocessing.native import save_sample, subsample_pcd, downsample_pcd
        from icgs.geometry.transforms import transform_pcd
        self.validate_context(context)
        horizon = self.graph_config.pred_horizon
        points=observation.points
        if self.runtime.live_voxel_size is not None:
            points=downsample_pcd(points,self.runtime.live_voxel_size)
        live = dict(obs=[transform_pcd(subsample_pcd(points), np.linalg.inv(observation.T_w_e))],
                    grips=[observation.grip], actions_grip=[np.zeros(horizon)],
                    T_w_es=[observation.T_w_e],
                    actions=[observation.T_w_e.reshape(1,4,4).repeat(horizon, axis=0)])
        data = save_sample({'demos': context.demos, 'live': live}, None).to(self.runtime.device)
        if not self.runtime.cache_context:
            # Published vv19 public path: noise is drawn before first forward scene encoding.
            return self.predict_batch(data)
        # Keep original pre-sampling cache computation outside no_grad/autocast and its RNG order.
        if context.embeddings is None:
            context.embeddings, context.positions = self.network.get_demo_scene_emb(data)
        data.live_scene_node_embds, data.live_scene_node_pos = self.network.get_live_scene_emb(data)
        data.live_scene_node_embds = data.live_scene_node_embds.clone()
        data.live_scene_node_pos = data.live_scene_node_pos.clone()
        data.demo_scene_node_embds = context.embeddings.clone()
        data.demo_scene_node_pos = context.positions.clone()
        return self.predict_batch(data)
