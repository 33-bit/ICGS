def run(args):
    from icgs.cli.observability import close_cli_run, start_cli_run
    from icgs.configuration.loader import load_config
    from icgs.artifacts.published import load_published_policy,published_config
    from pathlib import Path
    from icgs.environments.rlbench.adapter import RLBenchAdapter
    from icgs.evaluation.runner import evaluate_policy
    from dataclasses import replace
    recorder = start_cli_run(args, "evaluate", metadata={"device": args.device})
    status = "succeeded"
    error = None
    try:
        with recorder.span("config.load", component="cli"):
            config=load_config(args.config,entry='eval',overrides={'runtime.device':args.device,
                                'graph.num_demos':args.num_demos,'evaluation.num_rollouts':args.num_rollouts})
        # This CLI's checkpoint route is the strict published profile only.
        expected=published_config(device=config.runtime.device,num_demos=config.graph.num_demos,diffusion_steps=config.sampling.steps)
        for name in ('scene','graph','backbone','action','diffusion','runtime','sampling'):
            actual=getattr(config,name); target=getattr(expected,name)
            if name=='scene': actual=replace(actual,checkpoint=target.checkpoint)
            if actual!=target: raise ValueError(f'evaluate published checkpoint has unsupported architecture override: {name}')
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'model.pt'
        with recorder.span("artifacts.load", component="cli"):
            policy=load_published_policy(args.checkpoint,device=config.runtime.device,num_demos=config.graph.num_demos,diffusion_steps=config.sampling.steps)
        recorder.write_resolved_config(
            config,
            metadata={
                'identities': {
                    'checkpoint_sha256': policy.artifact_sha256,
                    'checkpoint_path': str(checkpoint_path),
                },
                'unavailable': {
                    'dataset': 'evaluation environment supplies task data; no dataset manifest at this boundary',
                    'reference': 'published loader does not expose a separate reference manifest',
                },
            },
        )
        e=config.evaluation
        env=RLBenchAdapter(e.task_name,headless=e.headless,restrict_rot=e.restrict_rot)
        result=evaluate_policy(policy,env,num_demos=config.graph.num_demos,num_rollouts=e.num_rollouts,
                               max_execution_steps=e.max_execution_steps,execution_horizon=e.execution_horizon,
                               num_traj_wp=config.graph.traj_horizon, recorder=recorder)
        print('Success rate:',result)
    except BaseException as exc:
        error = exc
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        close_cli_run(recorder, status=status, error=error)
