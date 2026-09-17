"""Bounded headless renderer/physics readiness check, NOT ICGS training data.

Worker runs in the provisioned Python 3.11 environment. The Colab launcher invokes
it with xvfb-run and a hard subprocess timeout; all outputs are under preflight/.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def worker(output):
    import numpy as np
    from pyrep import PyRep
    from pyrep.backend import sim as sim_api
    from pyrep.objects.vision_sensor import VisionSensor
    from rlbench import environment as rl_environment
    from rlbench.backend.const import TTT_FILE

    output = Path(output)
    sim = PyRep()
    launched = False
    report = {'kind': 'simulator_readiness_not_training_data', 'training_episodes': 0}
    try:
        scene = Path(rl_environment.__file__).parent / TTT_FILE
        sim.launch(str(scene), headless=True)
        launched = True
        sim.start()
        before = sim_api.simGetSimulationTime()
        dt = sim.get_simulation_timestep()
        for _ in range(2):
            sim.step()
        after = sim_api.simGetSimulationTime()
        camera = VisionSensor('cam_front')
        # The upstream base scene renders this camera automatically on sim.step.
        depth = camera.capture_depth(in_meters=True)
        points = camera.pointcloud_from_depth(depth)
        np.savez_compressed(
            output / 'simulator-readiness.npz', depth_m=depth, points_w=points,
            camera_matrix=camera.get_matrix(), intrinsics=camera.get_intrinsic_matrix(),
        )
        if not np.isfinite(points).all() or not after > before:
            raise RuntimeError('Nonfinite rendered cloud or nonadvancing simulator clock')
        report.update({
            'status': 'PASS', 'physics_steps': 2, 'physics_dt_s': dt,
            'simulator_before_s': before, 'simulator_after_s': after,
            'point_cloud_shape': list(points.shape), 'depth_shape': list(depth.shape),
            'timed_control_certified': False, 'task_predicates_certified': False,
            'scene_source': 'upstream RLBench base scene, no task episode',
        })
    except BaseException as exc:
        report.update({'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        if launched:
            sim.shutdown()
        (output / 'simulator-readiness.json').write_text(json.dumps(report, indent=2) + '\n')


def launch():
    receipt = json.loads(Path('/content/icgs_storage_receipt.json').read_text())
    output = Path(receipt['drive_file']).parent
    history = output / 'attempt-history'
    history.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for name in ('simulator-readiness.log', 'simulator-readiness.json'):
        old = output / name
        if old.exists():
            old.rename(history / (stamp + '-' + name))
    env = os.environ.copy()
    sim_root = '/content/icgs-simulator/CoppeliaSim'
    env['COPPELIASIM_ROOT'] = sim_root
    env['LD_LIBRARY_PATH'] = sim_root + ':' + env.get('LD_LIBRARY_PATH', '')
    env['QT_QPA_PLATFORM_PLUGIN_PATH'] = sim_root
    env['QT_QPA_PLATFORM'] = 'xcb'
    env['QT_LOGGING_RULES'] = '*.debug=false'
    args = [
        'xvfb-run', '-a', '-s', '-screen 0 1280x1024x24',
        '/content/icgs-data-env/bin/python', '-B',
        '/content/colab_simulator_probe.py', '--worker', str(output),
    ]
    with (output / 'simulator-readiness.log').open('w') as log:
        result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    print('Simulator readiness process returncode:', result.returncode)
    report_path = output / 'simulator-readiness.json'
    if report_path.exists():
        print(report_path.read_text())
    if result.returncode:
        print((output / 'simulator-readiness.log').read_text()[-6000:])
        raise RuntimeError('Simulator readiness failed, logs retained on Drive')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        launch()
