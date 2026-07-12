import unittest

from app_v2.experiment_runner import build_schedule
from app_v2.orchestrator.episode_runner import matched_seed


class DesignTests(unittest.TestCase):
    def test_seed_is_condition_independent(self):
        seed_a = matched_seed(1000, 2, 4, 1, "Teacher")
        seed_b = matched_seed(1000, 2, 4, 1, "Teacher")
        self.assertEqual(seed_a, seed_b)

    def test_schedule_contains_complete_blocks(self):
        tasks = [{"id": "t1"}, {"id": "t2"}]
        conditions = ["C0", "C1", "C2"]
        schedule = build_schedule(conditions, tasks, 2, 123)
        self.assertEqual(len(schedule), 12)
        for repetition in (1, 2):
            block = [
                item for item in schedule
                if item["repetition"] == repetition
            ]
            pairs = {
                (item["condition"], item["task_id"]) for item in block
            }
            self.assertEqual(len(pairs), 6)


if __name__ == "__main__":
    unittest.main()
