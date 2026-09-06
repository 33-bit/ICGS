"""Lightning owns logging/optimization/checkpoint lifecycle, not policy mathematics."""
import lightning as L
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from ip.geometry import rotation_matrix_to_angle_axis
from ip.utils.repairs import repair_checkpoint
from ip.configs.original import from_legacy, to_legacy


class GraphDiffusion(L.LightningModule):
    def __init__(self, config, policy=None):
        super().__init__()
        from ip.composition import build_policy
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
        from ip.checkpoints import load_checkpoint_state
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
        from ip.configs.original import resolved_config
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


def run_training(model, resolved, data_path_train, data_path_val, *, use_wandb=False, run_name='test'):
    """Run the original Lightning training procedure on an already composed model.

    This launches potentially very long training; call only with explicit data/compute scope.
    The historical record=True/use_wandb=False logger defect is not corrected here.
    """
    import os
    import pickle
    import json
    from torch_geometric.data import DataLoader
    from lightning.pytorch.callbacks import LearningRateMonitor
    from lightning.pytorch.loggers import WandbLogger
    from ip.data.dataset import RunningDataset
    from ip.configs.original import resolved_config
    config = to_legacy(resolved)
    record, save_dir = config['record'], config['save_dir']
    if record and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dset_val = RunningDataset(data_path_val, len(os.listdir(data_path_val)), rand_g_prob=0)
    dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False)

    dset = RunningDataset(data_path_train, len(os.listdir(data_path_train)), rand_g_prob=config['randomize_g_prob'])
    dataloader = DataLoader(dset, batch_size=config['batch_size'], drop_last=True, shuffle=True,
                            num_workers=8, pin_memory=True)
    ####################################################################################################################
    if record:
        if use_wandb:
            logger = WandbLogger(project='Instant Policy',
                                 name=f'{run_name}',
                                 save_dir=save_dir,
                                 log_model=False)
        # Dump config to save_dir
        pickle.dump(config, open(f'{save_dir}/config.pkl', 'wb'))
        with open(f'{save_dir}/resolved_config.json', 'w') as stream:
            json.dump(resolved_config(resolved), stream, indent=2)
    else:
        logger = None
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        enable_checkpointing=False,  # We save the models manually.
        accelerator=config['device'],
        devices=1,
        max_steps=config['num_iters'],
        enable_progress_bar=True,
        precision='16-mixed',
        val_check_interval=20000,  # TODO: might want to change that.
        num_sanity_val_steps=2,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=500,  # TODO: might want to change that.
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
        callbacks=[lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader_val,
    )

    # Save last:
    if record:
        model.save_model(f'{save_dir}/last.pt')

    return model
