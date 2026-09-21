from pathlib import Path


TASK_DIR = Path("artifacts/g2-pilot/tasks")
TASK_MODULES = (
    "t06_place_into_drawer.py",
    "t08_pack_two_objects.py",
    "t09_place_into_holder.py",
    "t11_rotate_and_place.py",
    "t13_park_blocker_retrieve.py",
    "t14_park_and_restore.py",
)


def test_pilot_task_sources_use_approved_position_tolerance():
    for name in TASK_MODULES:
        source = (TASK_DIR / name).read_text(encoding="utf-8")
        assert "0.08" not in source
        assert "0.1))" not in source
        assert "0.01" in source
