"""Lightning owns logging/optimization/checkpoint lifecycle, not policy mathematics."""
import lightning as L
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from icgs.geometry.transforms import rotation_matrix_to_angle_axis
from icgs.artifacts.legacy_formats import repair_checkpoint
from icgs.configuration.defaults import from_legacy, to_legacy


class GraphDiffusion(L.LightningModule):
    def __init__(self, config, policy=None):
        super().__init__()
        from icgs.composition import build_policy
        self.resolved_config = from_legacy(config)
        self.policy = policy if policy is not None else build_policy(self.resolved_config)
        self.model = self.policy.network
        config = to_legacy(self.resolved_config)
        self.graph_rep = self.model.graph
        self.scene_encoder = self.model.scene_encoder
        self.local_encoder = self.model.local_encoder
        self.cond_encoder = self.model.cond_encoder
        self.action_encoder = self.model.action_encoder
        self.action_head_trans = self.model.prediction_head
        self.action_head_rot = self.model.prediction_head_rot
        self.action_head_grip = self.model.prediction_head_g
        ################################################################################################################
        self.config = config
        self.record = config['record']
        self.save_dir = config['save_dir']
        self.save_every = config['save_every']
        self.randomise_num_demos = config['randomise_num_demos']
        self.use_lr_scheduler = config['use_lr_scheduler']
        self.best_trans_loss = 1e6
        self.best_sr = 0
        self.val_losses = []


        self.noise_scheduler = getattr(self.policy.objective, 'noise_scheduler', None)
        self.normalizer = self.model.codec
        self.objective = self.policy.objective

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *, config, strict=True, map_location=None, **kwargs):
        from icgs.artifacts.checkpoints import load_checkpoint_state
        if kwargs:
            raise TypeError(f"Unsupported legacy load options: {sorted(kwargs)}")
        model = cls(config)
        model.checkpoint_report = load_checkpoint_state(
            checkpoint_path, model, strict=strict,
            map_location=map_location or model.resolved_config.runtime.device)
        return model

    def add_noise(self, actions, grip_actions, timesteps):
        return self.objective.add_noise(actions, grip_actions, timesteps)

    def se3_loss(self, pred, gt):
        '''
        pred: (B, T, 4, 4)
        gt: (B, T, 4, 4)
        '''
        # Get the translation and rotation components
        trans_err = torch.norm(pred[..., :3, 3] - gt[..., :3, 3], dim=-1).mean()
        rot_error = torch.eye(4, device=pred.device, dtype=pred.dtype).repeat(pred.shape[0], pred.shape[1], 1, 1)
        rot_error[..., :3, :3] = pred[..., :3, :3].transpose(-1, -2) @ gt[..., :3, :3]
        rot_error = rot_error.view(-1, 4, 4)
        angle_axis = rotation_matrix_to_angle_axis(rot_error[:, :3, :])
        rot_error = angle_axis.norm(dim=-1).mean() * 180 / np.pi
        return trans_err, rot_error


    def training_step(self, data, batch_idx):
        if self.randomise_num_demos:
            num_demos = np.random.randint(1, self.config['num_demos'] + 1)
            self.model.reinit_graphs(data.actions.shape[0], num_demos=num_demos)
        loss = self.objective(self.model, data)
        self.log("Train_Loss", loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx, vis=False, ret_actions=False):
        batch_size = data.actions.shape[0]
        self.model.reinit_graphs(batch_size, num_demos=self.config['num_demos_test'])
        gt_actions, gt_grips = data.actions.clone(), data.actions_grip.clone()

        with torch.autocast(dtype=torch.float32, device_type='cuda'):  # Need to be f32 for SVD.
            actions, grips = self.test_step(data, batch_idx, vis=vis)

        grip_loss = (grips.squeeze() - gt_grips.squeeze()).abs().mean()
        trans_err, rot_error = self.se3_loss(actions, gt_actions)
        self.log("Val_Grip_Loss", grip_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Val_Trans_Loss", trans_err, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Val_Rot_Loss", rot_error, on_step=False, on_epoch=True, prog_bar=True)

        self.model.reinit_graphs(self.config['batch_size'], num_demos=self.config['num_demos'])
        self.val_losses.append(trans_err)
        if ret_actions:
            return actions, grips
        loss = 0
        return loss


    def test_step(self, data, batch_idx=0, vis=False):
        result = self.policy.sampler.sample(self.model, data)
        return result.transforms, result.grips

    def on_save_checkpoint(self, checkpoint):
        from icgs.configuration.defaults import resolved_config
        checkpoint['instant_policy_config'] = resolved_config(self.resolved_config)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                      weight_decay=self.config['weight_decay'])
        if self.use_lr_scheduler:
            lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=self.config['num_warmup_steps'],
                num_training_steps=self.config['num_iters'],
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        return optimizer

    def on_validation_epoch_end(self, *args, **kwargs):
        mean_trans_err = torch.tensor(self.val_losses).mean()
        self.val_losses = []
        if self.best_trans_loss > mean_trans_err and self.record:
            # TODO: Could be smarter.
            self.save_model(f'{self.save_dir}/best.pt')
            self.best_trans_loss = mean_trans_err

    def save_model(self, path, save_compiled=False):
        self.trainer.save_checkpoint(path)
        if self.config['compile_models']:
            repair_checkpoint(path, save_path=path)
            if save_compiled:
                path_compiled = path.replace('.pt', '_compiled.pt')
                self.trainer.save_checkpoint(path_compiled)

    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step % self.save_every == 0 and self.record:
            self.save_model(f'{self.save_dir}/{self.global_step}.pt', save_compiled=False)

        # TODO: Can run evals and log results here.

    def on_train_epoch_end(self, *args, **kwargs):
        if self.record:
            self.save_model(f'{self.save_dir}/last.pt', save_compiled=True)
