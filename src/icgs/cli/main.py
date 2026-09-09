import argparse


def _add_logging_args(parser):
    parser.add_argument('--logging-config', help='Explicit logging-only JSON envelope')
    parser.add_argument('--log-dir', help='Override the local observability run root')
    parser.add_argument('--log-level', help='Set local console and file level')
    parser.add_argument('--trace-mode', choices=('off', 'normal', 'debug', 'capture'),
                        help='Set local observability mode')


def main(argv=None):
    parser=argparse.ArgumentParser(prog='icgs',description='Unified ICGS research runtime')
    commands=parser.add_subparsers(dest='command',required=True)
    infer=commands.add_parser('infer',help='Strict published-checkpoint inference from version1 NPZ')
    infer.add_argument('--checkpoint',required=True)
    infer.add_argument('--input',required=True)
    infer.add_argument('--output',required=True)
    infer.add_argument('--device',default='cuda')
    infer.add_argument('--num-demos',type=int,default=2)
    infer.add_argument('--diffusion-steps',type=int,default=4)
    infer.add_argument('--seed',type=int,default=None)
    _add_logging_args(infer)
    train=commands.add_parser('train',help='Run native diffusion training on existing PyG data')
    train.add_argument('--config',required=True)
    train.add_argument('--data-train',required=True)
    train.add_argument('--data-val',required=True)
    train.add_argument('--checkpoint')
    train.add_argument('--device')
    train.add_argument('--run-name',default='test')
    train.add_argument('--use-wandb',action='store_true')
    _add_logging_args(train)
    evaluate=commands.add_parser('evaluate',help='Run explicitly requested RLBench evaluation')
    evaluate.add_argument('--config',required=True)
    evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--device')
    evaluate.add_argument('--num-demos',type=int)
    evaluate.add_argument('--num-rollouts',type=int)
    _add_logging_args(evaluate)
    prepare=commands.add_parser('prepare-data',help='Convert a supplied raw-demo/live NPZ into native PyG data')
    prepare.add_argument('--config',required=True)
    prepare.add_argument('--input',required=True)
    prepare.add_argument('--output',required=True)
    prepare.add_argument('--device')
    prepare.add_argument('--offset',type=int,default=0)
    prepare.add_argument('--cache-embeddings',action='store_true')
    _add_logging_args(prepare)
    logs=commands.add_parser('logs',help='Inspect local observability runs without loading models')
    log_commands=logs.add_subparsers(dest='logs_command',required=True)
    inspect=log_commands.add_parser('inspect',help='Summarize one local run')
    inspect.add_argument('run_dir')
    inspect.add_argument('--json',action='store_true')
    trace=log_commands.add_parser('trace',help='Print correlated spans and events')
    trace.add_argument('run_dir')
    trace.add_argument('--episode')
    trace.add_argument('--decision')
    trace.add_argument('--candidate')
    trace.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='infer':
        from icgs.cli.infer import run
        run(args)
    elif args.command=='train':
        from icgs.cli.train import run
        run(args)
    elif args.command=='evaluate':
        from icgs.cli.evaluate import run
        run(args)
    elif args.command=='prepare-data':
        from icgs.cli.prepare_data import run
        run(args)
    elif args.command=='logs':
        from icgs.cli.logs import run_inspect, run_trace
        if args.logs_command == 'inspect':
            run_inspect(args)
        else:
            run_trace(args)


if __name__=='__main__': main()
