from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from app_v2.memory.base import BaseMemory, MemoryRead, bounded_recent


class PerAgentMemory(BaseMemory):
    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.data: DefaultDict[
            str, DefaultDict[str, List[Dict[str, Any]]]
        ] = defaultdict(lambda: defaultdict(list))

    def get_context(
        self, episode_id: str, agent_name: str, query: str, round_idx: int
    ) -> MemoryRead:
        fragments = [
            item
            for item in self.data.get(episode_id, {}).get(agent_name, [])
            if int(item["round_idx"]) < round_idx
        ]
        parts = [
            f"[Earlier round {item['round_idx'] + 1} | {agent_name}]\n"
            f"{item['text']}"
            for item in fragments
        ]
        text = bounded_recent(parts, self.max_chars)
        return MemoryRead(
            text=text,
            visibility="private_per_agent",
            fragment_count=len(fragments),
            context_chars=len(text),
            source_agents=[agent_name] if fragments else [],
            source_rounds=sorted(
                {int(item["round_idx"]) for item in fragments}
            ),
            cross_agent_fragments=0,
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
            self.data[episode_id][agent_name].append(
                {"round_idx": round_idx, "text": final_output.strip()}
            )

    def snapshot(self, episode_id: str) -> Dict[str, Any]:
        groups = self.data.get(episode_id, {})
        fragments = [item for values in groups.values() for item in values]
        return {
            "visibility": "private_per_agent",
            "stored_fragments": len(fragments),
            "stored_chars": sum(len(item["text"]) for item in fragments),
            "stored_agent_groups": len(groups),
            "stores_agent_outputs": True,
            "stores_reasoning": False,
        }

    def clear_episode(self, episode_id: str) -> bool:
        self.data.pop(episode_id, None)
        return episode_id not in self.data
