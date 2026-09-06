"""Original objective and sampler. Arithmetic/order retained; no training framework."""
import torch
from icgs.contracts.records import ActionTrajectory


def original_scheduler(config):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    return DDIMScheduler(num_train_timesteps=config.train_steps,
                         beta_schedule='squaredcos_cap_v2', clip_sample=False,
                         prediction_type='sample')


class OriginalDiffusionObjective:
    def __init__(self, config, codec, scheduler):
        self.config, self.codec, self.noise_scheduler = config, codec, scheduler
        self.loss_f = torch.nn.L1Loss()

    def add_noise(self, actions, grip_actions, timesteps):
        '''
        actions: (B, T, 4, 4)
        grip_actions: (B, T, 1)
        timesteps: (B,)
        '''
        # First convert 4x4 to 6 (translation + angle axis)
        b, t = actions.shape[:2]

        actions_6d = self.codec.encode(actions.view(-1, 4, 4)).view(b, t, 6)
        # Normalize the actions
        actions_6d = self.codec.normalize_actions(actions_6d)
        # Add noise
        noise = torch.randn(actions_6d.shape, device=actions.device, dtype=actions_6d.dtype)
        noisy_actions = self.noise_scheduler.add_noise(actions_6d, noise, timesteps)
        noisy_actions = torch.clamp(noisy_actions, -1, 1)
        # Denormalize the actions
        noisy_actions = self.codec.denormalize_actions(noisy_actions)
        # Convert back to 4x4
        noisy_actions = self.codec.decode(noisy_actions.view(-1, 6)).view(b, t, 4, 4)

        # Add noise to the gripper actions
        noise_g = torch.randn(grip_actions.shape, device=actions.device, dtype=grip_actions.dtype)
        noisy_grip_actions = self.noise_scheduler.add_noise(grip_actions, noise_g, timesteps)
        noisy_grip_actions = torch.clamp(noisy_grip_actions, -1, 1)

        return noisy_actions, noisy_grip_actions


    def __call__(self, network, data):
        batch_size = data.actions.shape[0]
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(0, self.config.train_steps,
                                  (batch_size,), device=data.actions.device).long()
        noisy_actions, noisy_grip_actions = self.add_noise(data.actions, data.actions_grip, timesteps)

        labels = self.codec.get_labels(data.actions, noisy_actions,
                                       data.actions_grip.unsqueeze(-1), noisy_grip_actions.unsqueeze(-1),
                                       delta_grip=False)

        labels[..., :6] = self.codec.normalize_labels(labels[..., :6])

        # Store the noisy actions and grips in the data object as gt actions.
        data.actions = noisy_actions
        data.actions_grip = noisy_grip_actions
        data.diff_time = timesteps.view(-1, 1)
        preds = network(data)

        loss = self.loss_f(preds, labels)

        return loss



class OriginalDiffusionSampler:
    def __init__(self, config, diffusion, codec, scheduler):
        self.config, self.diffusion, self.codec = config, diffusion, codec
        self.noise_scheduler = scheduler

    def sample(self, network, data):
        batch_size = data.actions.shape[0]
        noisy_actions = torch.randn(
            (batch_size, self.codec.pred_horizon, 6), device=data.actions.device
        )
        noisy_actions = torch.clamp(noisy_actions, -1, 1)
        noisy_actions = self.codec.denormalize_actions(noisy_actions)
        noisy_actions = self.codec.decode(noisy_actions.view(-1, 6)).view(batch_size, -1, 4, 4)

        noisy_grips = torch.randn((batch_size, self.codec.pred_horizon, 1), device=data.actions.device)
        noisy_grips = torch.clamp(noisy_grips, -1, 1)

        # init scheduler
        self.noise_scheduler.set_timesteps(self.config.steps)

        for k in range(self.config.steps - 1, -1, -1):

            data.actions = noisy_actions
            data.actions_grip = noisy_grips.squeeze(-1)
            data.diff_time = torch.tensor([[
                k if k != self.config.steps - 1 else self.diffusion.train_steps
            ]] * batch_size, device=data.actions.device)

            preds = network(data)
            preds[..., :6] = self.codec.denormalize_labels(preds[..., :6])

            current_gripper_pos = self.codec.get_transformed_node_pos(noisy_actions, transform=False)
            mode_output = preds[..., 3:6] + current_gripper_pos + torch.mean(preds[..., :3], dim=-2, keepdim=True)

            # Diffusion step for the actions
            pred_girpper_pos = self.noise_scheduler.step(
                model_output=mode_output,
                sample=current_gripper_pos,
                timestep=k,
            ).prev_sample

            # Get the transformation matrices for the gripper
            T_e_e = self.codec.rigid_alignment(current_gripper_pos.view(-1, pred_girpper_pos.shape[-2], 3),
                                         pred_girpper_pos.view(-1,
                                                               pred_girpper_pos.shape[-2], 3)).view(batch_size,
                                                                                                    -1,
                                                                                                    4, 4)

            noisy_actions = torch.matmul(noisy_actions, T_e_e)

            # Diffusion step for the gripper
            noisy_grips = self.noise_scheduler.step(
                model_output=preds[..., -1:].mean(dim=-2),  # + noisy_grips,
                sample=noisy_grips,
                timestep=k,
            ).prev_sample
            noisy_grips = torch.clamp(noisy_grips, -1, 1)

            # Convert to 6d, normalize, clamp and denormalize
            noisy_actions_6d = self.codec.encode(noisy_actions.view(-1, 4, 4)).view(batch_size, -1, 6)
            noisy_actions_6d = self.codec.normalize_actions(noisy_actions_6d)
            noisy_actions_6d = torch.clamp(noisy_actions_6d, -1, 1)
            noisy_actions_6d = self.codec.denormalize_actions(noisy_actions_6d)
            noisy_actions = self.codec.decode(noisy_actions_6d.view(-1, 6)).view(batch_size, -1, 4, 4)
            noisy_actions = noisy_actions.view(batch_size, -1, 4, 4)

        return ActionTrajectory(noisy_actions, torch.sign(noisy_grips))
