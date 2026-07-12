from __future__ import annotations

from typing import Any, Dict, List, Optional

from app_v2.agents.base import BaseAgent
from app_v2.llm_client import LLMClient


SYSTEM_PROMPT = (
    "You are an internal quality-control evaluator. Assess only the current "
    "learner-facing answer against the learning goal, learner profile, current "
    "learner event, and adaptation target. You have no access to longitudinal "
    "memory. Return only valid JSON matching the requested schema. This is an "
    "internal diagnostic and not a substitute for independent human review."
)


class EvaluatorAgent(BaseAgent):
    def __init__(self, client: LLMClient):
        super().__init__("Evaluator", SYSTEM_PROMPT, client)

    def response_format(self) -> Optional[Dict[str, Any]]:
        score = {"type": "integer", "minimum": 1, "maximum": 5}
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "clarity": score,
                    "correctness": score,
                    "pedagogical_alignment": score,
                    "responsiveness": score,
                    "continuity": score,
                    "trustworthiness": score,
                    "failure_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "brief_comment": {
                        "type": "string",
                        "maxLength": 500,
                    },
                },
                "required": [
                    "clarity",
                    "correctness",
                    "pedagogical_alignment",
                    "responsiveness",
                    "continuity",
                    "trustworthiness",
                    "failure_flags",
                    "brief_comment",
                ],
                "additionalProperties": False,
            },
        }

    def build_messages(
        self,
        task: Dict[str, Any],
        round_data: Dict[str, Any],
        memory_context: str,
        upstream_output: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if not upstream_output:
            raise ValueError("Evaluator requires the current Adapter answer")

        profile = task["learner_profile"]
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Learning goal:\n{task['learning_goal']}\n\n"
                    "Learner profile:\n"
                    f"- Prior knowledge: {profile['prior_knowledge']}\n"
                    f"- Preferences: {profile['preferences']}\n"
                    f"- Communication needs: {profile['communication']}\n\n"
                    f"Current learner event:\n{round_data['learner_event']}\n\n"
                    f"Adaptation target:\n{round_data['adaptation_target']}\n\n"
                    f"Answer to evaluate:\n{upstream_output}\n\n"
                    "Score every dimension from 1 to 5. Use failure_flags for "
                    "factual errors, ignored constraints, unsupported claims, "
                    "unsafe advice, or obvious truncation."
                ),
            },
        ]
