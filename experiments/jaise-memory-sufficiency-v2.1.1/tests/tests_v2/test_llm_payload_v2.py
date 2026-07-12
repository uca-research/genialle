import unittest

from app_v2.llm_client import LLMClient


class FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "OK",
                        "reasoning_content": "brief reasoning",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            "timings": {},
        }


class FakeSession:
    def __init__(self):
        self.payload = None

    def post(self, url, json, timeout):
        self.payload = json
        return FakeResponse()


class LLMRequestPayloadTests(unittest.TestCase):
    def test_gpt_oss_reasoning_effort_is_sent(self):
        settings = {
            "model": {
                "backend_url": "http://127.0.0.1:8033/v1",
                "name": "gpt-oss-20b",
                "request_timeout_sec": 10,
                "max_retries": 0,
                "retry_backoff_sec": 0,
                "reasoning_format": "auto",
                "cache_prompt": False,
                "agent_generation": {
                    "Evaluator": {
                        "temperature": 0.0,
                        "max_tokens": 640,
                        "reasoning_effort": "low",
                    }
                },
            }
        }
        client = LLMClient(settings)
        fake = FakeSession()
        client.session = fake

        client.call(
            [{"role": "user", "content": "Test"}],
            agent_name="Evaluator",
            seed=123,
        )

        self.assertEqual(
            fake.payload["chat_template_kwargs"],
            {"reasoning_effort": "low"},
        )
        self.assertNotIn(
            "enable_thinking",
            fake.payload["chat_template_kwargs"],
        )
        self.assertEqual(fake.payload["max_tokens"], 640)
        self.assertEqual(fake.payload["seed"], 123)


if __name__ == "__main__":
    unittest.main()
