"""Training orchestration, separate from mathematical objectives."""
from icgs.configuration.defaults import to_legacy


def build_training_logger(*, record, use_wandb, recorder=None, run_name='test'):
    """Construct the outer logger without constructing datasets or a Trainer."""
    from dataclasses import replace
    from icgs.observability.config import default_config
    from icgs.observability.lightning import RecorderLightningLogger
    from icgs.observability.recorder import NoopRecorder
    from icgs.observability.wandb import start as start_wandb

    recorder = recorder or NoopRecorder()
    if not record:
        return None, None
    remote = None
    if use_wandb:
        wandb_config = getattr(getattr(recorder, 'config', None), 'wandb', None)
        if wandb_config is None:
            wandb_config = replace(default_config().wandb, enabled=True)
        elif not wandb_config.enabled:
            wandb_config = replace(wandb_config, enabled=True)
        remote = start_wandb(wandb_config, run_id=recorder.run_id or run_name,
                             metadata={'run_name': run_name}, recorder=recorder)
    return RecorderLightningLogger(recorder, remote=remote), remote


def run_training(model, resolved, data_path_train, data_path_val, *, use_wandb=False, run_name='test', recorder=None):
    """Run the original Lightning training procedure on an already composed model.

    This launches potentially very long training; call only with explicit data/compute scope.
    The historical record=True/use_wandb=False logger defect is handled by an
    explicit local logger; no training job is started by logger construction.
    """
    import os
    import pickle
    import json
    import lightning as L
    from torch_geometric.data import DataLoader
    from lightning.pytorch.callbacks import LearningRateMonitor
    from icgs.observability.recorder import NoopRecorder
    from icgs.data.datasets.native import RunningDataset
    from icgs.configuration.defaults import resolved_config
    config = to_legacy(resolved)
    recorder = recorder or NoopRecorder()
    record, save_dir = config['record'], config['save_dir']
    if record and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger, remote = build_training_logger(record=record, use_wandb=use_wandb,
                                           recorder=recorder, run_name=run_name)
    dset_val = RunningDataset(data_path_val, len(os.listdir(data_path_val)), rand_g_prob=0)
    dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False)

    dset = RunningDataset(data_path_train, len(os.listdir(data_path_train)), rand_g_prob=config['randomize_g_prob'])
    dataloader = DataLoader(dset, batch_size=config['batch_size'], drop_last=True, shuffle=True,
                            num_workers=8, pin_memory=True)
    ####################################################################################################################
    if record:
        # Dump config to save_dir
        pickle.dump(config, open(f'{save_dir}/config.pkl', 'wb'))
        with open(f'{save_dir}/resolved_config.json', 'w') as stream:
            json.dump(resolved_config(resolved), stream, indent=2)
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
        callbacks=[lr_monitor] if logger is not None else [],
    )

    try:
        with recorder.span("training.fit", component="training"):
            trainer.fit(
                model=model,
                train_dataloaders=dataloader,
                val_dataloaders=dataloader_val,
            )

        # Save last:
        if record:
            with recorder.span("training.checkpoint", component="training"):
                model.save_model(f'{save_dir}/last.pt')
    finally:
        if remote is not None:
            remote.close()

    return model
