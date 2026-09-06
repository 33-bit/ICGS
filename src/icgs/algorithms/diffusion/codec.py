from icgs.geometry.transforms import transforms_to_actions, actions_to_transforms, get_rigid_transforms
import torch


class Normalizer:
    def __init__(self, pred_horizon, min_action, max_action, gripper_length=0.06, device='cuda'):
        self.pred_horizon = pred_horizon

        self.min_action = min_action[None, None, :].repeat(1, pred_horizon, 1)
        self.max_action = max_action[None, None, :].repeat(1, pred_horizon, 1)

        # Assumes that min_action are negative and max_action are positive
        # Scale 1 dim by the corresponding time step
        self.min_action *= torch.linspace(1, pred_horizon, pred_horizon, device=device, dtype=torch.float)[None, :,
                           None]
        self.max_action *= torch.linspace(1, pred_horizon, pred_horizon, device=device, dtype=torch.float)[None, :,
                           None]

        max_angle = self.max_action[..., -1]
        delta_rot = torch.sqrt(2 * gripper_length ** 2 - 2 * gripper_length ** 2 * torch.cos(max_angle))
        delta_rot = delta_rot[..., None, None].repeat(1, 1, 1, 3)

        self.min_labels = (2 * self.min_action[..., :3]).unsqueeze(2)
        self.max_labels = (2 * self.max_action[..., :3]).unsqueeze(2)

        self.min_labels = torch.cat([self.min_labels[..., :3], -2 * delta_rot], dim=-1)
        self.max_labels = torch.cat([self.max_labels[..., :3], 2 * delta_rot], dim=-1)

    def normalize_actions(self, actions):
        '''
        Normalize actions to [-1, 1]
        :param actions : (bs, pred_horizon, 6)
        :return: normalized actions : (bs, pred_horizon, 6)
        '''
        return 2 * (actions - self.min_action) / (self.max_action - self.min_action) - 1

    def denormalize_actions(self, actions):
        '''
        Denormalize actions to the original range
        :param actions : (bs, pred_horizon, 6)
        :return: denormalized actions : (bs, pred_horizon, 6)
        '''
        return 0.5 * (actions + 1) * (self.max_action - self.min_action) + self.min_action

    def normalize_labels(self, labels):
        '''
        Normalize labels to [-1, 1]
        :param labels : (bs, pred_horizon, 3)
        :return: normalized labels : (bs, pred_horizon, 3)
        '''
        return 2 * (labels - self.min_labels) / (self.max_labels - self.min_labels) - 1

    def denormalize_labels(self, labels):
        '''
        Denormalize labels to the original range
        :param labels : (bs, pred_horizon, 3)
        :return: denormalized labels : (bs, pred_horizon, 3)
        '''
        return 0.5 * (labels + 1) * (self.max_labels - self.min_labels) + self.min_labels


class OriginalActionCodec(Normalizer):
    """Original SE(3)/gripper representation and gripper-node supervision."""

    def __init__(self, config, gripper_node_pos, device):
        super().__init__(config.pred_horizon,
                         torch.tensor(config.minimum, dtype=torch.float32, device=device),
                         torch.tensor(config.maximum, dtype=torch.float32, device=device), device=device)
        self.gripper_node_pos = gripper_node_pos

    def encode(self, transforms):
        return transforms_to_actions(transforms)

    def decode(self, actions):
        return actions_to_transforms(actions)

    def rigid_alignment(self, source, target):
        return get_rigid_transforms(source, target)

    def transform_gripper_nodes(self, gripper_nodes, T):
        # gripper_nodes - [B, D, T, N, 3]
        # T - [B, D, T, 4, 4]
        has_demo = len(gripper_nodes.shape) == 5
        if not has_demo:
            gripper_nodes = gripper_nodes.unsqueeze(1)
        b, d, t, n, _ = gripper_nodes.shape
        gripper_nodes = gripper_nodes.reshape(-1, gripper_nodes.shape[-2], gripper_nodes.shape[-1]).permute(0, 2, 1)
        gripper_nodes = torch.bmm(T[..., :3, :3].reshape(-1, 3, 3), gripper_nodes)
        gripper_nodes += T[..., :3, 3].reshape(-1, 3, 1)
        gripper_nodes = gripper_nodes.permute(0, 2, 1).view(b, d, t, n, 3)
        if not has_demo:
            gripper_nodes = gripper_nodes.squeeze(1)
        return gripper_nodes

    def get_labels(self, gt_actions, noisy_actions, gt_grips, noisy_grips, delta_grip=False, sep_rot=True):
        # gt_actions: (bs, pred_horizon, 4, 4)
        # noisy_actions: (bs, pred_horizon, 4, 4)
        # gt_grips: (bs, pred_horizon, 1)
        # noisy_grips: (bs, pred_horizon, 1)
        gripper_points = self.gripper_node_pos[None, None, :].repeat(gt_actions.shape[0],
                                                                           gt_actions.shape[1], 1, 1)

        if sep_rot:
            T_w_n = noisy_actions.view(-1, 4, 4)
            T_n_w = torch.inverse(T_w_n)
            T_w_g = gt_actions.view(-1, 4, 4)
            T_n_g = torch.bmm(T_n_w, T_w_g)
            T_n_g = T_n_g.view(gt_actions.shape[0], gt_actions.shape[1], 4, 4)

            labels_trans = T_n_g[..., :3, 3][:, :, None, :].repeat(1, 1,
                                                                   gripper_points.shape[-2],
                                                                   1)
            T_n_g[..., :3, 3] = 0
            labels_rot = self.transform_gripper_nodes(gripper_points, T_n_g) - gripper_points
            labels = torch.cat([labels_trans, labels_rot], dim=-1)
        else:
            gripper_points_gt = self.transform_gripper_nodes(gripper_points, gt_actions)
            gripper_points_noisy = self.transform_gripper_nodes(gripper_points, noisy_actions)
            labels = gripper_points_gt - gripper_points_noisy

        if delta_grip:
            labels_grip = gt_grips - noisy_grips
        else:
            labels_grip = gt_grips
        labels_grip = labels_grip[:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)
        labels = torch.cat([labels, labels_grip], dim=-1)
        return labels

    def get_transformed_node_pos(self, actions, transform=True):
        gripper_points = self.gripper_node_pos[None, None, :].repeat(actions.shape[0], actions.shape[1], 1, 1)
        if transform:
            gripper_points = self.transform_gripper_nodes(gripper_points, actions)
        return gripper_points
