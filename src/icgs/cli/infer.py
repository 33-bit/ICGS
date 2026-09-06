def run(args):
    import json
    import time
    import torch
    from icgs.artifacts.published import load_published_policy
    from icgs.data.schemas.inference import load_input,save_output
    from icgs.state.randomness import scoped_seed
    observation,demos=load_input(args.input)
    if len(demos)!=args.num_demos: raise ValueError('input demo count does not match requested session')
    policy=load_published_policy(args.checkpoint,device=args.device,num_demos=args.num_demos,diffusion_steps=args.diffusion_steps)
    start=time.perf_counter()
    with scoped_seed(args.seed,device=args.device),torch.no_grad():
        context=policy.prepare_context(demos)
        result=policy.predict(observation,context)
    if torch.device(args.device).type=='cuda': torch.cuda.synchronize()
    metadata=dict(checkpoint_sha256=policy.artifact_sha256,seed=args.seed,device=args.device,
                  num_demos=args.num_demos,diffusion_steps=args.diffusion_steps,seconds=time.perf_counter()-start,
                  frames='actions=root-relative; absolute_targets=root_pose @ actions',
                  input_kind='caller-supplied; no task-success claim')
    save_output(args.output,result,observation,metadata=metadata)
    print(json.dumps(dict(status='PASS',**metadata)),flush=True)
