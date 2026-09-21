
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import Condition
from rlbench.backend.task import Task


class NearCondition(Condition):
    def __init__(self, obj, target, tolerance):
        self.obj = obj
        self.target = target
        self.tolerance = float(tolerance)

    def condition_met(self):
        distance = float(np.linalg.norm(self.obj.get_position() - self.target.get_position()))
        return distance <= self.tolerance, False


class ProceduralTask(Task):
    OBJECTS = ()
    CONDITIONS = ()
    WAYPOINT_COUNT = 0
    DESCRIPTION = ''

    def init_task(self):
        self._objects = {name: Shape(name) for name in self.OBJECTS if name != 'target_a'}
        self._targets = {name: Shape(name) for name in self.OBJECTS if name.startswith('target') or name.endswith('target')}
        self._handles = {**self._objects, **self._targets}
        graspable = [obj for name, obj in self._objects.items() if name.startswith('object') or name == 'blocker']
        if graspable:
            self.register_graspable_objects(graspable)
        conditions = [NearCondition(self._handles[a], self._handles[b], tol) for a, b, tol in self.CONDITIONS]
        self.register_success_conditions(conditions)

    def init_episode(self, index):
        self._variation_index = int(index)
        # Deterministic small translation/yaw variation; the manifest records the range.
        dx = ((index * 17) % 7 - 3) * 0.004
        dy = ((index * 29) % 7 - 3) * 0.004
        for name, obj in self._objects.items():
            pos = obj.get_position()
            obj.set_position([float(pos[0] + dx), float(pos[1] + dy), float(pos[2])])
        return [self.DESCRIPTION]

    def variation_count(self):
        return 32

    def is_static_workspace(self):
        return True

    def validate(self):
        # The pilot expert is a direct pose controller; waypoint metadata is
        # retained for provenance but RLBench's optional waypoint feasibility
        # preflight is not used as the execution path.
        self._waypoints = []


class T14ParkAndRestore(ProceduralTask):
    OBJECTS = ('blocker', 'object_a', 'park_target', 'restore_target')
    CONDITIONS = (('blocker', 'restore_target', 0.01),)
    WAYPOINT_COUNT = 8
    DESCRIPTION = 'park blocker and restore it after a retrieval motion'
