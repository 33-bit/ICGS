"""RLBench boundary; importing this module does not import or launch RLBench."""

from collections.abc import Mapping

import numpy as np
from tqdm import tqdm

from icgs.data.preprocessing.native import downsample_pcd, sample_to_cond_demo
from icgs.execution.rollout import evaluate_policy
from icgs.geometry.transforms import pose_to_transform, transform_to_pose
from icgs.contracts.records import Observation


class _RLBenchTaskMapping(Mapping):
    def __init__(self, class_names):
        self._class_names = class_names

    def __getitem__(self, key):
        from rlbench import tasks
        return getattr(tasks, self._class_names[key])

    def __iter__(self):
        return iter(self._class_names)

    def __len__(self):
        return len(self._class_names)

    def __contains__(self, key):
        return key in self._class_names


TASK_NAMES = _RLBenchTaskMapping({
    'lift_lid': 'TakeLidOffSaucepan',
    'phone_on_base': 'PhoneOnBase',
    'open_box': 'OpenBox',
    'slide_block': 'SlideBlockToTarget',
    'close_box': 'CloseBox',
    'basketball': 'BasketballInHoop',
    'buzz': 'BeatTheBuzz',
    'close_microwave': 'CloseMicrowave',
    'plate_out': 'TakePlateOffColoredDishRack',
    'toilet_seat_down': 'ToiletSeatDown',
    'toilet_seat_up': 'ToiletSeatUp',
    'toilet_roll_off': 'TakeToiletRollOffStand',
    'open_microwave': 'OpenMicrowave',
    'lamp_on': 'LampOn',
    'umbrella_out': 'TakeUmbrellaOutOfUmbrellaStand',
    'push_button': 'PushButton',
    'put_rubbish': 'PutRubbishInBin',
})


def override_bounds(pos, rot, env):
    from rlbench.backend.spawn_boundary import BoundingBox
    if pos is not None:
        BoundingBox.within_boundary = lambda x, y, z: True
        env._scene._workspace_boundary._boundaries[0]._get_position_within_boundary = lambda x, y: pos
    env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, rot - 0.0001),
                                                     (0.0, 0.0, rot + 0.0001))


def get_point_cloud(obs, camera_names=('front', 'left_shoulder', 'right_shoulder')):
    pcds = []
    for camera_name in camera_names:
        ordered_pcd = getattr(obs, f'{camera_name}_point_cloud')
        mask = getattr(obs, f'{camera_name}_mask')
        masked_pcd = ordered_pcd[mask > 60]
        pcds.append(masked_pcd)
    return downsample_pcd(np.concatenate(pcds, axis=0))


def rl_bench_demo_to_sample(demo):
    sample = {'pcds': [], 'T_w_es': [], 'grips': []}
    for obs in demo:
        sample['pcds'].append(get_point_cloud(obs))
        sample['T_w_es'].append(pose_to_transform(obs.gripper_pose))
        sample['grips'].append(obs.gripper_open)
    return sample


class RLBenchAdapter:
    def __init__(self, task_name, headless=False, restrict_rot=True):
        self.task_name = task_name
        self.headless = headless
        self.restrict_rot = restrict_rot
        self.env = None
        self.task = None

    def launch(self):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig

        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(),
            gripper_action_mode=Discrete(),
        )
        self.env = Environment(action_mode, './', obs_config=obs_config,
                               headless=self.headless)
        self.env.launch()
        self.task = self.env.get_task(TASK_NAMES[self.task_name])

        def temp(position, euler=None, quaternion=None, ignore_collisions=False,
                 trials=300, max_configs=1, distance_threshold=0.65,
                 max_time_ms=10, trials_per_goal=1, algorithm=None,
                 relative_to=None):
            return self.env._robot.arm.get_linear_path(
                position, euler, quaternion, ignore_collisions=ignore_collisions,
                relative_to=relative_to)

        self.env._robot.arm.get_path = temp
        self.env._scene._start_arm_joint_pos = np.array([
            6.74760377e-05, -1.91104114e-02, -3.62065766e-05,
            -1.64271665e+00, -1.14094291e-07, 1.55336857e+00,
            7.85427451e-01,
        ])
        rot_bounds = self.env._scene.task.base_rotation_bounds()
        mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
        if self.restrict_rot:
            self.env._scene.task.base_rotation_bounds = lambda: (
                (0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)),
            )

    def collect_demos(self, num_demos, num_traj_wp):
        demos = [dict()] * num_demos
        for i in tqdm(range(num_demos), desc='Collecting demos', total=num_demos, leave=False):
            done = False
            while not done:
                try:
                    rlbench_demos = self.task.get_demos(1, live_demos=True, max_attempts=1000)
                    sample = rl_bench_demo_to_sample(rlbench_demos[0])
                    demos[i] = sample_to_cond_demo(sample, num_traj_wp)
                    assert len(demos[i]['obs']) == num_traj_wp
                    done = True
                except:
                    continue
        return demos

    def reset(self):
        done = False
        while not done:
            try:
                self.task.reset()
                done = True
            except:
                continue

    def observe(self):
        observation = self.task.get_observation()
        return Observation(points=get_point_cloud(observation),
                           T_w_e=pose_to_transform(observation.gripper_pose),
                           grip=observation.gripper_open)

    def encode_action(self, observation, trajectory, index):
        actions = trajectory.transforms.squeeze(0).cpu().numpy()
        grips = trajectory.grips.squeeze(0).squeeze(-1).cpu().numpy()
        command = np.zeros(8)
        command[:7] = transform_to_pose(observation.T_w_e @ actions[index])
        command[7] = int((grips[index] + 1) / 2 > 0.5)
        return command

    def step(self, command):
        _, reward, terminate = self.task.step(command)
        return reward, terminate

    def shutdown(self):
        self.env.shutdown()


def rollout_model(model, num_demos, task_name='phone_on_base', max_execution_steps=30,
                  execution_horizon=8, num_rollouts=2, headless=False,
                  num_traj_wp=10, restrict_rot=True):
    if hasattr(model, 'predict'):
        policy = model
    elif hasattr(model, 'policy'):
        policy = model.policy
    else:
        raise TypeError('rollout_model requires an InstantPolicy or a wrapper with a policy attribute')
    environment = RLBenchAdapter(task_name, headless=headless, restrict_rot=restrict_rot)
    return evaluate_policy(policy, environment, num_demos=num_demos,
                           num_rollouts=num_rollouts,
                           max_execution_steps=max_execution_steps,
                           execution_horizon=execution_horizon,
                           num_traj_wp=num_traj_wp)
