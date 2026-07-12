from __future__ import annotations

from typing import Any, Dict, List, Optional

from app_v2.agents.base import BaseAgent
from app_v2.llm_client import LLMClient


SYSTEM_PROMPT = (
    "You are the Adapter agent in a human-centred intelligent learning "
    "environment. Revise the Teacher draft to fit the learner profile, the "
    "current learner event, and the adaptation target. Preserve technical "
    "correctness and resolve the learner's stated difficulty. Use supplied "
    "memory only for continuity with earlier rounds. Do not expose hidden "
    "reasoning or discuss the multi-agent workflow. Return only the final "
    "learner-facing response. Aim for 220 to 350 words unless less is needed."
)


class AdapterAgent(BaseAgent):
    def __init__(self, client: LLMClient):
        super().__init__("Adapter", SYSTEM_PROMPT, client)

    def build_messages(
        self,
        task: Dict[str, Any],
        round_data: Dict[str, Any],
        memory_context: str,
        upstream_output: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if not upstream_output:
            raise ValueError("Adapter requires the current Teacher draft")

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
                    f"Adaptation target:\n{round_data['adaptation_target']}\n\n"
                    f"Current Teacher draft:\n{upstream_output}\n\n"
                    "Rewrite the draft once. Do not add an evaluation rubric."
                ),
            }
        )
        return messages
