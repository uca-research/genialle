from __future__ import annotations

from typing import Any, Dict

from app_v2.memory.base import BaseMemory, MemoryRead


class NoMemory(BaseMemory):
    def get_context(
        self, episode_id: str, agent_name: str, query: str, round_idx: int
    ) -> MemoryRead:
        return MemoryRead()

    def update(
        self,
        episode_id: str,
        agent_name: str,
        round_idx: int,
        final_output: str,
        metadata: Dict[str, Any],
    ) -> None:
        return None

    def snapshot(self, episode_id: str) -> Dict[str, Any]:
        return {
            "visibility": "none",
            "stored_fragments": 0,
            "stored_chars": 0,
            "stores_agent_outputs": False,
            "stores_reasoning": False,
        }

    def clear_episode(self, episode_id: str) -> bool:
        return True
