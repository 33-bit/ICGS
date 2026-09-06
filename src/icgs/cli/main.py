import argparse


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
    train=commands.add_parser('train',help='Run native diffusion training on existing PyG data')
    train.add_argument('--config',required=True)
    train.add_argument('--data-train',required=True)
    train.add_argument('--data-val',required=True)
    train.add_argument('--checkpoint')
    train.add_argument('--device')
    train.add_argument('--run-name',default='test')
    train.add_argument('--use-wandb',action='store_true')
    evaluate=commands.add_parser('evaluate',help='Run explicitly requested RLBench evaluation')
    evaluate.add_argument('--config',required=True)
    evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--device')
    evaluate.add_argument('--num-demos',type=int)
    evaluate.add_argument('--num-rollouts',type=int)
    prepare=commands.add_parser('prepare-data',help='Convert a supplied raw-demo/live NPZ into native PyG data')
    prepare.add_argument('--config',required=True)
    prepare.add_argument('--input',required=True)
    prepare.add_argument('--output',required=True)
    prepare.add_argument('--device')
    prepare.add_argument('--offset',type=int,default=0)
    prepare.add_argument('--cache-embeddings',action='store_true')
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


if __name__=='__main__': main()
