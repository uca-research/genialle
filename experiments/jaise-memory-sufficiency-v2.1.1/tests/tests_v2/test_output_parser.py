import unittest

from app_v2.output_parser import extract_final_content, parse_json_object


class OutputParserTests(unittest.TestCase):
    def test_harmony_final(self):
        raw = (
            "<|channel|>analysis<|message|>hidden"
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            "Visible answer<|end|>"
        )
        final, markers = extract_final_content(raw)
        self.assertEqual(final, "Visible answer")
        self.assertTrue(markers)

    def test_plain_final(self):
        final, markers = extract_final_content("Visible answer")
        self.assertEqual(final, "Visible answer")
        self.assertFalse(markers)

    def test_json_fence(self):
        data = parse_json_object('```json\n{"clarity": 5}\n```')
        self.assertEqual(data["clarity"], 5)


if __name__ == "__main__":
    unittest.main()
