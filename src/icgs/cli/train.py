def run(args):
    from icgs.configuration.loader import load_config
    from icgs.composition import build_training_module
    from icgs.artifacts.checkpoints import load_checkpoint_state
    from icgs.training.runner import run_training
    config=load_config(args.config,entry='train',overrides={'runtime.device':args.device})
    module=build_training_module(config)
    if args.checkpoint:
        load_checkpoint_state(args.checkpoint,module,strict=True,map_location=config.runtime.device)
    return run_training(module,config,args.data_train,args.data_val,use_wandb=args.use_wandb,run_name=args.run_name)
