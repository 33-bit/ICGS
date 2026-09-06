def run(args):
    """Convert explicit raw-demo NPZ and a live trajectory; no fabricated data collector."""
    from pathlib import Path
    from icgs.configuration.loader import load_config
    from icgs.data.schemas.inference import load_input
    from icgs.data.preprocessing.native import sample_to_cond_demo,sample_to_live,save_sample
    config=load_config(args.config,overrides={'runtime.device':args.device})
    _,demonstrations=load_input(args.input)
    # Input holds D conditioning demos followed by one live training trajectory.
    if len(demonstrations)!=config.graph.num_demos+1:
        raise ValueError('prepare-data needs D conditioning demos plus one live trajectory')
    demos=[sample_to_cond_demo(d,config.graph.traj_horizon) for d in demonstrations[:-1]]
    live=sample_to_live(demonstrations[-1],config.graph.pred_horizon,2048,.01,3,subsample=False)
    encoder=None
    if args.cache_embeddings:
        from icgs.composition import build_scene_encoder
        encoder=build_scene_encoder(config.scene,device=config.runtime.device).eval()
    Path(args.output).mkdir(parents=True,exist_ok=True)
    save_sample({'demos':demos,'live':live},args.output,offset=args.offset,scene_encoder=encoder)
