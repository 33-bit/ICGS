"""Example only: supply robot observations/demos/controller before running."""
from ip.composition import load_policy
from ip.types import Observation


if __name__ == '__main__':
    num_demos = 2
    num_diffusion_iters = 4
    compile_models = False
    max_execution_steps = 100

    # Same composition and inference path as evaluation; legacy non-strict mode reports mismatches.
    policy = load_policy('./checkpoints', mode='deploy', num_demos=num_demos,
                         compile_models=compile_models)
    # TODO: collect {'pcds': [...], 'T_w_es': [...], 'grips': [...]} demonstrations.
    demos = []
    context = policy.prepare_context(demos)

    for k in range(max_execution_steps):
        # TODO: segmented world XYZ [N,3], end-effector-to-world [4,4], grip 0/1.
        pcd_w, T_w_e, grip = None, None, None
        trajectory = policy.predict(Observation(pcd_w, T_w_e, grip), context)
        actions = trajectory.transforms.squeeze(0).cpu().numpy()
        grips = trajectory.grips.squeeze(0).cpu().numpy()
        # TODO: execute all/part of this observation-relative chunk with your controller.
        # World target j is T_w_e @ actions[j]; never chain relative actions.
        # -1 closes the gripper; +1 opens it. No robot controller is supplied.
