import unittest

from app_v2.memory import (
    LexicalRetrievalMemory,
    NoMemory,
    PerAgentMemory,
    ResponsibleLearnerStateMemory,
    SharedMemory,
)


class MemoryTests(unittest.TestCase):
    def exercise_agent_output_memory(self, memory):
        episode = "episode"
        memory.start_episode(episode, {})
        memory.update(episode, "Teacher", 0, "first-round output", {})
        same_round = memory.get_context(
            episode, "Adapter", "first-round output", 0
        )
        self.assertNotIn("first-round output", same_round.text)
        self.assertFalse(same_round.contains_current_round)

        next_round = memory.get_context(
            episode, "Teacher", "first-round output", 1
        )
        self.assertFalse(next_round.contains_current_round)
        self.assertTrue(memory.clear_episode(episode))

    def test_c0(self):
        self.exercise_agent_output_memory(NoMemory())

    def test_c1(self):
        self.exercise_agent_output_memory(PerAgentMemory(2000))

    def test_c2(self):
        self.exercise_agent_output_memory(SharedMemory(2000))

    def test_c3(self):
        self.exercise_agent_output_memory(
            LexicalRetrievalMemory(3, 2000)
        )

    def test_c4_role_based_access_and_deletion(self):
        memory = ResponsibleLearnerStateMemory(2000)
        episode = "episode"
        memory.start_episode(episode, {})
        memory.record_round_state(
            episode,
            0,
            {
                "learner_event": "I need a concrete example.",
                "adaptation_target": "Use an example.",
            },
        )
        teacher = memory.get_context(episode, "Teacher", "example", 1)
        evaluator = memory.get_context(episode, "Evaluator", "example", 1)
        self.assertIn("I need a concrete example.", teacher.text)
        self.assertEqual(evaluator.text, "")
        self.assertFalse(teacher.contains_current_round)
        self.assertTrue(memory.clear_episode(episode))


if __name__ == "__main__":
    unittest.main()
