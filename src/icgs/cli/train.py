def run(args):
    from icgs.cli.observability import close_cli_run, start_cli_run
    from icgs.configuration.loader import load_config
    from icgs.composition import build_training_module
    from icgs.artifacts.checkpoints import load_checkpoint_state
    from icgs.training.runner import run_training
    recorder = start_cli_run(args, "train", metadata={"device": args.device, "run_name": args.run_name},
                             enable_wandb=args.use_wandb)
    status = "succeeded"
    error = None
    try:
        with recorder.span("config.load", component="cli"):
            config=load_config(args.config,entry='train',overrides={'runtime.device':args.device})
        recorder.write_resolved_config(
            config,
            metadata={
                'identities': {
                    'config_path': str(args.config),
                    'data_train_path': str(args.data_train),
                    'data_val_path': str(args.data_val),
                    **({'checkpoint_path': str(args.checkpoint)} if args.checkpoint else {}),
                },
                'unavailable': {
                    'dataset_sha256': 'native dataset loader identity is not exposed at this boundary',
                    'checkpoint_sha256': 'checkpoint loader identity is not exposed at this boundary',
                },
            },
        )
        with recorder.span("composition.build", component="cli"):
            module=build_training_module(config)
        if args.checkpoint:
            with recorder.span("checkpoint.load", component="artifacts"):
                load_checkpoint_state(args.checkpoint,module,strict=True,map_location=config.runtime.device)
        return run_training(module,config,args.data_train,args.data_val,use_wandb=args.use_wandb,
                            run_name=args.run_name, recorder=recorder)
    except BaseException as exc:
        error = exc
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        close_cli_run(recorder, status=status, error=error)
