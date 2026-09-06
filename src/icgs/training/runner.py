"""Training orchestration, separate from mathematical objectives."""
import lightning as L
from icgs.configuration.defaults import to_legacy


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
    from icgs.data.datasets.native import RunningDataset
    from icgs.configuration.defaults import resolved_config
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
