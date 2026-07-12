from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from app_v2.memory.base import BaseMemory, MemoryRead, bounded_recent


class SharedMemory(BaseMemory):
    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.data: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    def get_context(
        self, episode_id: str, agent_name: str, query: str, round_idx: int
    ) -> MemoryRead:
        fragments = [
            item
            for item in self.data.get(episode_id, [])
            if int(item["round_idx"]) < round_idx
        ]
        parts = [
            f"[Earlier round {item['round_idx'] + 1} | {item['agent']}]\n"
            f"{item['text']}"
            for item in fragments
        ]
        text = bounded_recent(parts, self.max_chars)
        return MemoryRead(
            text=text,
            visibility="shared_blackboard",
            fragment_count=len(fragments),
            context_chars=len(text),
            source_agents=sorted({str(item["agent"]) for item in fragments}),
            source_rounds=sorted(
                {int(item["round_idx"]) for item in fragments}
            ),
            cross_agent_fragments=sum(
                1 for item in fragments if item["agent"] != agent_name
            ),
            contains_current_round=any(
                int(item["round_idx"]) >= round_idx for item in fragments
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
        if final_output:
            self.data[episode_id].append(
                {
                    "round_idx": round_idx,
                    "agent": agent_name,
                    "text": final_output.strip(),
                }
            )

    def snapshot(self, episode_id: str) -> Dict[str, Any]:
        fragments = self.data.get(episode_id, [])
        return {
            "visibility": "shared_blackboard",
            "stored_fragments": len(fragments),
            "stored_chars": sum(len(item["text"]) for item in fragments),
            "stored_agent_groups": len(
                {str(item["agent"]) for item in fragments}
            ),
            "stores_agent_outputs": True,
            "stores_reasoning": False,
        }

    def clear_episode(self, episode_id: str) -> bool:
        self.data.pop(episode_id, None)
        return episode_id not in self.data
