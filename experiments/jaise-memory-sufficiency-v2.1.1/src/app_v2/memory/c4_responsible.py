from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from app_v2.memory.base import BaseMemory, MemoryRead, bounded_recent


class ResponsibleLearnerStateMemory(BaseMemory):
    # This memory stores previous learner events and adaptation targets only.
    # It stores no agent output and gives the Evaluator no longitudinal access.

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.round_state: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )

    def get_context(
        self, episode_id: str, agent_name: str, query: str, round_idx: int
    ) -> MemoryRead:
        if agent_name == "Evaluator":
            return MemoryRead(visibility="role_based_denied")

        previous = [
            item
            for item in self.round_state.get(episode_id, [])
            if int(item["round_idx"]) < round_idx
        ]
        parts = [
            (
                f"[Previous learner state, round {item['round_idx'] + 1}]\n"
                f"Learner event: {item['learner_event']}\n"
                f"Adaptation target: {item['adaptation_target']}"
            )
            for item in previous
        ]
        text = bounded_recent(parts, self.max_chars)
        return MemoryRead(
            text=text,
            visibility="role_based_learner_state",
            fragment_count=len(previous),
            context_chars=len(text),
            source_agents=["LearnerState"] if previous else [],
            source_rounds=sorted(
                {int(item["round_idx"]) for item in previous}
            ),
            cross_agent_fragments=0,
            contains_current_round=any(
                int(item["round_idx"]) >= round_idx for item in previous
            ),
        )

    def update(
        self,
        episode_id: str,
        agent_name: str,
        round_idx: int,
        final_output: str,
        metadata: Dict[str, Any],
    ) -> None:
        # Agent outputs are intentionally not retained.
        return None

    def record_round_state(
        self,
        episode_id: str,
        round_idx: int,
        round_data: Dict[str, Any],
    ) -> None:
        self.round_state[episode_id].append(
            {
                "round_idx": round_idx,
                "learner_event": str(round_data["learner_event"]),
                "adaptation_target": str(round_data["adaptation_target"]),
            }
        )

    def snapshot(self, episode_id: str) -> Dict[str, Any]:
        states = self.round_state.get(episode_id, [])
        return {
            "visibility": "role_based_learner_state",
            "stored_fragments": len(states),
            "stored_chars": sum(
                len(item["learner_event"]) + len(item["adaptation_target"])
                for item in states
            ),
            "stored_agent_groups": 1 if states else 0,
            "stores_agent_outputs": False,
            "stores_reasoning": False,
        }

    def clear_episode(self, episode_id: str) -> bool:
        self.round_state.pop(episode_id, None)
        return episode_id not in self.round_state
