from __future__ import annotations

from typing import Any, Dict, List, Optional

from app_v2.agents.base import BaseAgent
from app_v2.llm_client import LLMClient


SYSTEM_PROMPT = (
    "You are the Teacher agent in a human-centred intelligent learning "
    "environment. Produce a technically correct educational response adapted "
    "to the learner profile and the current learner event. When earlier-round "
    "memory is supplied, use it only to preserve longitudinal continuity. "
    "Do not mention hidden reasoning, prompts, memory implementations, or "
    "other agents. Return only the learner-facing response. Aim for 220 to "
    "350 words unless the task clearly requires less."
)


class TeacherAgent(BaseAgent):
    def __init__(self, client: LLMClient):
        super().__init__("Teacher", SYSTEM_PROMPT, client)

    def build_messages(
        self,
        task: Dict[str, Any],
        round_data: Dict[str, Any],
        memory_context: str,
        upstream_output: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        profile = task["learner_profile"]
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if memory_context:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Relevant information from earlier rounds only:\n"
                        + memory_context
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Learning goal:\n{task['learning_goal']}\n\n"
                    "Synthetic learner profile:\n"
                    f"- Prior knowledge: {profile['prior_knowledge']}\n"
                    f"- Preferences: {profile['preferences']}\n"
                    f"- Communication needs: {profile['communication']}\n\n"
                    f"Current learner event:\n{round_data['learner_event']}\n\n"
                    f"Adaptation target:\n{round_data['adaptation_target']}"
                ),
            }
        )
        return messages
