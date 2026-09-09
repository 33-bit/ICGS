"""Policy/environment rollout orchestration independent of concrete models."""

from tqdm import trange
from icgs.contracts.records import PredictivePolicy, EvaluationEnvironment


def evaluate_policy(policy: PredictivePolicy, environment: EvaluationEnvironment, num_demos=2, num_rollouts=2,
                    max_execution_steps=30, execution_horizon=8, num_traj_wp=10, recorder=None):
    if recorder is None:
        from icgs.observability.recorder import NoopRecorder
        recorder = NoopRecorder()
    policy.eval()
    with recorder.span("environment.launch", component="evaluation"):
        environment.launch()
    with recorder.span("evaluation.collect_demos", component="evaluation",
                        fields={"num_demos": num_demos, "num_traj_wp": num_traj_wp}):
        demos = environment.collect_demos(num_demos, num_traj_wp)
    with recorder.span("evaluation.prepare_context", component="evaluation"):
        context = policy.prepare_context(demos, prepared=True)

    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    for rollout_index in pbar:
        bound = recorder.bind(episode_id=str(rollout_index), origin="real")
        with bound.span("evaluation.rollout", component="evaluation",
                        fields={"rollout_index": rollout_index}):
            environment.reset()
            policy.reset_context(context)
            success = 0
            for _ in range(max_execution_steps):
                observation = environment.observe()
                with bound.span("policy.predict", component="policy"):
                    trajectory = policy.predict(observation, context)
                for index in range(execution_horizon):
                    command = environment.encode_action(observation, trajectory, index)
                    try:
                        reward, terminate = environment.step(command)
                        success = int(terminate and reward > 0.)
                    except Exception as error:
                        bound.exception(error, component="environment",
                                        fields={"operation": "step", "index": index})
                        bound.event("environment.step.error", level="ERROR", component="environment",
                                    fields={"index": index})
                        terminate = True
                    if terminate:
                        break
                else:
                    continue
                break
            successes.append(success)
            bound.event("evaluation.rollout.end", component="evaluation",
                        fields={"rollout_index": rollout_index, "success": success})
            pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
            pbar.refresh()
    pbar.close()
    environment.shutdown()
    return sum(successes) / len(successes)
