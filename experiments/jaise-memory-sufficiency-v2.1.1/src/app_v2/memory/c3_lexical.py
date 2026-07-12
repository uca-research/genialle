from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple

from app_v2.memory.base import BaseMemory, MemoryRead, bounded_recent


def lexical_tokens(text: str) -> Set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", (text or "").lower()))


class LexicalRetrievalMemory(BaseMemory):
    def __init__(self, top_k: int, max_chars: int):
        self.top_k = top_k
        self.max_chars = max_chars
        self.data: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    def get_context(
        self, episode_id: str, agent_name: str, query: str, round_idx: int
    ) -> MemoryRead:
        query_tokens = lexical_tokens(query)
        candidates = [
            item
            for item in self.data.get(episode_id, [])
            if int(item["round_idx"]) < round_idx
        ]
        ranked: List[Tuple[int, int, Dict[str, Any]]] = []
        for item in candidates:
            score = len(query_tokens & item["tokens"])
            ranked.append((score, int(item["sequence"]), item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = [row[2] for row in ranked[: self.top_k]]
        selected.sort(
            key=lambda item: (int(item["round_idx"]), int(item["sequence"]))
        )

        parts = [
            (
                f"[Retrieved round {item['round_idx'] + 1} | "
                f"{item['agent']} | lexical_score="
                f"{len(query_tokens & item['tokens'])}]\n{item['text']}"
            )
            for item in selected
        ]
        text = bounded_recent(parts, self.max_chars)
        return MemoryRead(
            text=text,
            visibility="shared_lexical_retrieval",
            fragment_count=len(selected),
            context_chars=len(text),
            source_agents=sorted({str(item["agent"]) for item in selected}),
            source_rounds=sorted(
                {int(item["round_idx"]) for item in selected}
            ),
            cross_agent_fragments=sum(
                1 for item in selected if item["agent"] != agent_name
            ),
            contains_current_round=any(
                int(item["round_idx"]) >= round_idx for item in selected
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
            items = self.data[episode_id]
            items.append(
                {
                    "round_idx": round_idx,
                    "agent": agent_name,
                    "text": final_output.strip(),
                    "tokens": lexical_tokens(final_output),
                    "sequence": len(items),
                }
            )

    def snapshot(self, episode_id: str) -> Dict[str, Any]:
        fragments = self.data.get(episode_id, [])
        return {
            "visibility": "shared_lexical_retrieval",
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
