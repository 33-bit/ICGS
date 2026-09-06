"""Graph denoising network with injected components and original parameter names."""
import torch
from torch_geometric.utils import to_dense_batch
from ip.models.backbone import run_stages


class GraphDenoiser(torch.nn.Module):
    def __init__(self, graph_config, backbone_config, runtime, components, codec):
        super().__init__()
        self.num_demos = graph_config.num_demos
        self.num_demos_in_use = graph_config.num_demos
        self.traj_horizon = graph_config.traj_horizon
        self.local_embd_dim = graph_config.embd_dim
        self.batch_size = runtime.batch_size
        self.num_scenes_nodes = graph_config.num_scene_nodes
        self.pred_horizon = graph_config.pred_horizon
        self.num_layers = backbone_config.num_layers
        for name, module in components.items():
            setattr(self, name, module)
        self.codec = codec

    def reinit_graphs(self, batch_size, num_demos=None):
        self.batch_size = batch_size
        if num_demos is not None:
            self.num_demos = num_demos
            self.graph.num_demos = num_demos
        self.graph.batch_size = batch_size
        self.graph.initialise_graph()

    def compile_models(self):
        self.scene_encoder.sa1_module.conv = torch.compile(self.scene_encoder.sa1_module.conv, mode="reduce-overhead")
        self.scene_encoder.sa2_module.conv = torch.compile(self.scene_encoder.sa2_module.conv, mode="reduce-overhead")
        self.local_encoder = torch.compile(self.local_encoder, mode="reduce-overhead")
        self.cond_encoder = torch.compile(self.cond_encoder, mode="reduce-overhead")
        self.action_encoder = torch.compile(self.action_encoder, mode="reduce-overhead")
        self.prediction_head = torch.compile(self.prediction_head, mode="reduce-overhead")
        self.prediction_head_rot = torch.compile(self.prediction_head_rot, mode="reduce-overhead")
        self.prediction_head_g = torch.compile(self.prediction_head_g, mode="reduce-overhead")

    def get_labels(self, *args, **kwargs):
        return self.codec.get_labels(*args, **kwargs)

    def get_transformed_node_pos(self, actions, transform=True):
        return self.codec.get_transformed_node_pos(actions, transform)

    def forward(self, data):
        if not hasattr(data, 'demo_scene_node_embds'):
            data.demo_scene_node_embds, data.demo_scene_node_pos = self.get_demo_scene_emb(data)

        if not hasattr(data, 'live_scene_node_embds'):
            data.live_scene_node_embds, data.live_scene_node_pos = self.get_live_scene_emb(data)
        ################################################################################################################
        current_obs = to_dense_batch(data.pos_obs, data.batch_pos_obs, fill_value=0)[0]
        current_obs = current_obs[:, None, ...].repeat(1, self.pred_horizon, 1, 1)
        current_obs = current_obs.view(self.batch_size * self.pred_horizon, -1, 3)
        actions = data.actions.view(-1, 4, 4)

        current_obs = torch.bmm(actions[:, :3, :3].transpose(1, 2), current_obs.permute(0, 2, 1)).permute(0, 2, 1)
        current_obs -= actions[:, :3, 3][:, None, :]

        action_batch = torch.arange(current_obs.shape[0], device=current_obs.device)[:, None].repeat(1,
                                                                                                     current_obs.shape[
                                                                                                         1])
        action_batch = action_batch.view(-1)
        current_obs = current_obs.reshape(-1, 3)

        pos_obs_old = data.pos_obs.clone()
        batch_pos_obs_old = data.batch_pos_obs.clone()

        data.pos_obs = current_obs
        data.batch_pos_obs = action_batch

        action_scene_node_embds, action_scene_node_pos = self.get_live_scene_emb(data)

        data.pos_obs = pos_obs_old
        data.batch_pos_obs = batch_pos_obs_old

        data.action_scene_node_embds = action_scene_node_embds.view(self.batch_size, self.pred_horizon,
                                                                    -1, self.local_embd_dim)
        data.action_scene_node_pos = action_scene_node_pos.view(self.batch_size, self.pred_horizon, -1, 3)
        ################################################################################################################
        self.graph.update_graph(data)

        # TODO: This can cause problems for some GPU types when compiling, but it is needed for other types of GPUs.
        torch.compiler.cudagraph_mark_step_begin()

        x_dict = run_stages(self.local_encoder, self.cond_encoder, self.action_encoder, self.graph.graph)
        ################################################################################################################
        x_gripper = x_dict['gripper'][self.graph.graph.gripper_time > self.traj_horizon].view(self.batch_size,
                                                                                              self.pred_horizon,
                                                                                              self.graph.num_g_nodes,
                                                                                              -1)

        preds_t = self.prediction_head(x_gripper)
        preds_rot = self.prediction_head_rot(x_gripper)
        preds_g = self.prediction_head_g(x_gripper)
        preds = torch.cat([preds_t, preds_rot, preds_g], dim=-1)
        return preds

    def get_demo_scene_emb(self, data):
        bs = data.actions.shape[0]
        demo_scene_node_embds, demo_scene_node_pos, demo_scene_node_batch = \
            self.scene_encoder(None,
                               data.pos_demos,
                               data.batch_demos)
        demo_scene_node_embds = to_dense_batch(demo_scene_node_embds, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_embds = demo_scene_node_embds.view(bs, self.num_demos, self.traj_horizon, -1,
                                                           self.local_embd_dim)
        demo_scene_node_pos = to_dense_batch(demo_scene_node_pos, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_pos = demo_scene_node_pos.view(bs, self.num_demos, self.traj_horizon, -1, 3)
        return demo_scene_node_embds, demo_scene_node_pos

    def get_live_scene_emb(self, data):
        current_scene_node_embds, current_scene_node_pos, current_scene_node_batch = \
            self.scene_encoder(None,
                               data.pos_obs,
                               data.batch_pos_obs)
        current_scene_node_embds = to_dense_batch(current_scene_node_embds, current_scene_node_batch, fill_value=0)[0]
        current_scene_node_pos = to_dense_batch(current_scene_node_pos, current_scene_node_batch, fill_value=0)[0]
        return current_scene_node_embds, current_scene_node_pos
