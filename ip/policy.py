"""Public inference facade with per-policy, episode-local demonstration state."""
import torch
from ip.types import PreparedContext, Observation, ActionTrajectory


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
        if len(demos) != self.graph_config.num_demos:
            raise ValueError('demonstration count does not match policy graph')
        if not prepared:
            from ip.data.preprocessing import sample_to_cond_demo
            demos = [sample_to_cond_demo(demo, self.graph_config.traj_horizon) for demo in demos]
        # Do not run encoder early: original evaluation encodes on first predict per rollout.
        return PreparedContext(list(demos), self.context_owner)

    def validate_context(self, context):
        if context.owner is not self.context_owner:
            raise ValueError('context belongs to another policy; prepare it again')

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

    def predict(self, observation: Observation, context: PreparedContext) -> ActionTrajectory:
        import numpy as np
        from ip.data.preprocessing import save_sample, subsample_pcd
        from ip.geometry import transform_pcd
        self.validate_context(context)
        horizon = self.graph_config.pred_horizon
        live = dict(obs=[transform_pcd(subsample_pcd(observation.points), np.linalg.inv(observation.T_w_e))],
                    grips=[observation.grip], actions_grip=[np.zeros(horizon)],
                    T_w_es=[observation.T_w_e],
                    actions=[observation.T_w_e.reshape(1,4,4).repeat(horizon, axis=0)])
        data = save_sample({'demos': context.demos, 'live': live}, None).to(self.runtime.device)
        # Keep original pre-sampling cache computation outside no_grad/autocast and its RNG order.
        if context.embeddings is None:
            context.embeddings, context.positions = self.network.get_demo_scene_emb(data)
        data.live_scene_node_embds, data.live_scene_node_pos = self.network.get_live_scene_emb(data)
        data.live_scene_node_embds = data.live_scene_node_embds.clone()
        data.live_scene_node_pos = data.live_scene_node_pos.clone()
        data.demo_scene_node_embds = context.embeddings.clone()
        data.demo_scene_node_pos = context.positions.clone()
        return self.predict_batch(data)
