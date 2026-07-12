from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryRead:
    text: str = ""
    visibility: str = "none"
    fragment_count: int = 0
    context_chars: int = 0
    source_agents: List[str] = field(default_factory=list)
    source_rounds: List[int] = field(default_factory=list)
    cross_agent_fragments: int = 0
    contains_current_round: bool = False

    def metrics(self) -> Dict[str, Any]:
        result = asdict(self)
        result["context_sha256"] = (
            hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            if self.text
            else None
        )
        result.pop("text", None)
        return result


class BaseMemory(ABC):
    def start_episode(
        self, episode_id: str, learner_profile: Dict[str, str]
    ) -> None:
        return None

    @abstractmethod
    def get_context(
        self,
        episode_id: str,
        agent_name: str,
        query: str,
        round_idx: int,
    ) -> MemoryRead:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        episode_id: str,
        agent_name: str,
        round_idx: int,
        final_output: str,
        metadata: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def record_round_state(
        self,
        episode_id: str,
        round_idx: int,
        round_data: Dict[str, Any],
    ) -> None:
        return None

    @abstractmethod
    def snapshot(self, episode_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def clear_episode(self, episode_id: str) -> bool:
        raise NotImplementedError


def bounded_recent(parts: List[str], max_chars: int) -> str:
    selected: List[str] = []
    used = 0
    for part in reversed(parts):
        extra = len(part) + (2 if selected else 0)
        if selected and used + extra > max_chars:
            break
        if not selected and len(part) > max_chars:
            selected.append(part[-max_chars:])
            break
        selected.append(part)
        used += extra
    return "\n\n".join(reversed(selected))
