"""Policy/environment rollout orchestration independent of concrete models."""

from tqdm import trange
from ip.types import PredictivePolicy, EvaluationEnvironment


def evaluate_policy(policy: PredictivePolicy, environment: EvaluationEnvironment, num_demos=2, num_rollouts=2,
                    max_execution_steps=30, execution_horizon=8, num_traj_wp=10):
    policy.eval()
    environment.launch()
    demos = environment.collect_demos(num_demos, num_traj_wp)
    context = policy.prepare_context(demos, prepared=True)

    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    for _ in pbar:
        environment.reset()
        policy.reset_context(context)
        success = 0
        for _ in range(max_execution_steps):
            observation = environment.observe()
            trajectory = policy.predict(observation, context)
            for index in range(execution_horizon):
                command = environment.encode_action(observation, trajectory, index)
                try:
                    reward, terminate = environment.step(command)
                    success = int(terminate and reward > 0.)
                except Exception:
                    terminate = True
                if terminate:
                    break
            else:
                continue
            break
        successes.append(success)
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()
    pbar.close()
    environment.shutdown()
    return sum(successes) / len(successes)
