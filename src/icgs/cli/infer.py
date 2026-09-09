def run(args):
    import json
    from pathlib import Path
    import time
    import torch
    from icgs.cli.observability import close_cli_run, start_cli_run
    from icgs.artifacts.published import load_published_policy, published_config
    from icgs.data.schemas.inference import load_input,save_output
    from icgs.state.randomness import scoped_seed
    recorder = start_cli_run(args, "infer", metadata={"device": args.device, "seed": args.seed})
    status = "succeeded"
    error = None
    try:
        with recorder.span("input.validate", component="cli"):
            observation,demos=load_input(args.input)
            if len(demos)!=args.num_demos: raise ValueError('input demo count does not match requested session')
        with recorder.span("artifacts.load", component="cli"):
            checkpoint_path = Path(args.checkpoint).expanduser().resolve()
            if checkpoint_path.is_dir():
                checkpoint_path = checkpoint_path / "model.pt"
            policy=load_published_policy(args.checkpoint,device=args.device,num_demos=args.num_demos,diffusion_steps=args.diffusion_steps)
            recorder.write_resolved_config(
                published_config(device=args.device, num_demos=args.num_demos,
                                 diffusion_steps=args.diffusion_steps),
                metadata={
                    "identities": {
                        "checkpoint_sha256": policy.artifact_sha256,
                        "checkpoint_path": str(checkpoint_path),
                        "input_path": str(Path(args.input).expanduser().resolve()),
                    },
                    "unavailable": {
                        "dataset": "inference input is caller-supplied; no dataset manifest",
                        "reference": "published loader does not expose a separate reference manifest",
                    },
                },
            )
        start=time.perf_counter()
        with recorder.span("policy.inference", component="policy"):
            with scoped_seed(args.seed,device=args.device),torch.no_grad():
                with recorder.span("context.prepare", component="policy"):
                    context=policy.prepare_context(demos)
                with recorder.span("policy.predict", component="policy"):
                    result=policy.predict(observation,context)
        if torch.device(args.device).type=='cuda': torch.cuda.synchronize()
        metadata=dict(checkpoint_sha256=policy.artifact_sha256,seed=args.seed,device=args.device,
                      num_demos=args.num_demos,diffusion_steps=args.diffusion_steps,seconds=time.perf_counter()-start,
                      frames='actions=root-relative; absolute_targets=root_pose @ actions',
                      input_kind='caller-supplied; no task-success claim')
        with recorder.span("output.save", component="cli"):
            save_output(args.output,result,observation,metadata=metadata)
        print(json.dumps(dict(status='PASS',**metadata)),flush=True)
    except BaseException as exc:
        error = exc
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        close_cli_run(recorder, status=status, error=error)
