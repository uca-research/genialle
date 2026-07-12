from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app_v2.llm_client import LLMClient, LLMResult


class BaseAgent(ABC):
    def __init__(self, name: str, system_prompt: str, client: LLMClient):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client

    @abstractmethod
    def build_messages(
        self,
        task: Dict[str, Any],
        round_data: Dict[str, Any],
        memory_context: str,
        upstream_output: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        raise NotImplementedError

    def response_format(self) -> Optional[Dict[str, Any]]:
        return None

    def step(
        self,
        task: Dict[str, Any],
        round_data: Dict[str, Any],
        memory_context: str,
        seed: int,
        upstream_output: Optional[str] = None,
    ) -> LLMResult:
        return self.client.call(
            messages=self.build_messages(
                task=task,
                round_data=round_data,
                memory_context=memory_context,
                upstream_output=upstream_output,
            ),
            agent_name=self.name,
            seed=seed,
            response_format=self.response_format(),
        )
