from ip.composition import load_policy
from ip.utils.rl_bench_utils import rollout_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='plate_out')
    parser.add_argument('--num_demos', type=int, default=2)
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--restrict_rot', type=int, default=1)
    parser.add_argument('--compile_models', type=int, default=0)
    restrict_rot = bool(parser.parse_args().restrict_rot)
    task_name = parser.parse_args().task_name
    num_demos = parser.parse_args().num_demos
    num_rollouts = parser.parse_args().num_rollouts
    compile_models = bool(parser.parse_args().compile_models)
    ####################################################################################################################
    model_path = './checkpoints'
    policy = load_policy(model_path, mode='eval', num_demos=num_demos, compile_models=compile_models)
    sr = rollout_model(policy, num_demos, task_name, num_rollouts=num_rollouts, execution_horizon=8,
                       num_traj_wp=policy.graph_config.traj_horizon, restrict_rot=restrict_rot)
    print('Success rate:', sr)
