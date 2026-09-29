"""Pinned upstream identity and the native RoboHiMan collection protocol.

Pure data: importing this module does not import RoboHiMan, RLBench or PyRep.
The native split table transcribes ``HiMan-Bench/robot-colosseum/collect_dataset_*.sh``
at the pinned commit; ``scripts/robohiman_split_audit.py`` re-reads those files
and fails if this transcription drifts.
"""

from __future__ import annotations

from types import MappingProxyType


UPSTREAM = MappingProxyType({
    "robohiman": {"url": "https://github.com/chenyt31/RoboHiMan.git",
                  "revision": "33f71d30d9a100a83725ff00dfcc27374d5ce810"},
    "pyrep": {"url": "https://github.com/stepjam/PyRep.git",
              "revision": "231a1ac6b0a179cff53c1d403d379260b9f05f2f"},
    "rlbench": {"url": "https://github.com/buttomnutstoast/RLBench.git",
                "revision": "587a6a0e6dc8cd36612a208724eb275fe8cb4470"},
    "coppeliasim": {"version": "4.1.0", "archive": "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"},
})

ATOMIC_TASKS = (
    "box_in_cupboard", "box_out_of_opened_drawer", "close_drawer", "put_in_opened_drawer",
    "sweep_to_dustpan", "box_out_of_cupboard", "broom_out_of_cupboard", "open_drawer",
    "rubbish_in_dustpan", "take_out_of_opened_drawer",
)
COMPOSITIONAL_TRAIN_TASKS = ("put_in_without_close", "sweep_and_drop", "take_out_without_close", "transfer_box")
COMPOSITIONAL_TEST_TASKS = (
    "box_exchange", "put_in_and_close", "put_in_without_close", "put_two_in_different",
    "put_two_in_same", "retrieve_and_sweep", "sweep_and_drop", "take_out_and_close",
    "take_out_without_close", "take_two_out_of_different", "take_two_out_of_same", "transfer_box",
)

# name -> (script, tasks, generator family, idx_to_collect, episodes/task, image size, env seed)
NATIVE_SPLITS = MappingProxyType({
    "train_A": ("collect_dataset_train_A.sh", ATOMIC_TASKS, "atomic", 0, 20, (256, 256), 42),
    "train_AP": ("collect_dataset_train_AP.sh", ATOMIC_TASKS, "atomic", -1, 1, (256, 256), 42),
    "train_C": ("collect_dataset_train_C.sh", COMPOSITIONAL_TRAIN_TASKS, "compositional", 0, 5, (256, 256), 42),
    "train_CP": ("collect_dataset_train_CP.sh", COMPOSITIONAL_TRAIN_TASKS, "compositional", -1, 1, (256, 256), 42),
    "test_atomic": ("collect_dataset_test_atomic.sh", ATOMIC_TASKS, "atomic", -1, 1, (128, 128), 244),
    "test_compositional": ("collect_dataset_test_compositional.sh", COMPOSITIONAL_TEST_TASKS,
                           "compositional", -1, 1, (128, 128), 244),
})

# Native collection flags shared by every script at the pinned commit.
NATIVE_CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")


def task_family(task: str) -> str:
    if task in ATOMIC_TASKS:
        return "atomic"
    if task in COMPOSITIONAL_TEST_TASKS:
        return "compositional"
    raise KeyError(f"unknown RoboHiMan task {task!r}")


def level_for(task: str, strategy_index: int) -> str:
    """RoboHiMan level: strategy 0 is the unperturbed configuration."""
    atomic = task_family(task) == "atomic"
    if strategy_index == 0:
        return "A" if atomic else "C"
    return "AP" if atomic else "CP"


__all__ = [
    "ATOMIC_TASKS",
    "COMPOSITIONAL_TEST_TASKS",
    "COMPOSITIONAL_TRAIN_TASKS",
    "NATIVE_CAMERAS",
    "NATIVE_SPLITS",
    "UPSTREAM",
    "level_for",
    "task_family",
]
