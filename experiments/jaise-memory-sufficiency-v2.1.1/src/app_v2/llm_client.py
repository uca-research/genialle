from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from app_v2.output_parser import extract_final_content


@dataclass
class LLMResult:
    output: str
    usage: Dict[str, Any]
    timings: Dict[str, Any]
    finish_reason: Optional[str]
    latency_sec: float
    retry_count: int
    seed: int
    truncated: bool
    empty_output: bool
    raw_reasoning_markers: bool
    reasoning_chars: int
    raw_content_chars: int


class LLMClient:
    def __init__(self, settings: Dict[str, Any]):
        model_cfg = settings["model"]
        self.base_url = str(model_cfg["backend_url"]).rstrip("/")
        self.model_name = model_cfg["name"]
        self.timeout = float(model_cfg.get("request_timeout_sec", 180))
        self.max_retries = int(model_cfg.get("max_retries", 2))
        self.backoff = float(model_cfg.get("retry_backoff_sec", 1.0))
        self.reasoning_format = model_cfg.get("reasoning_format", "auto")
        self.cache_prompt = bool(model_cfg.get("cache_prompt", False))
        self.agent_generation = model_cfg["agent_generation"]
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        url = self.base_url.removesuffix("/v1") + "/health"
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        return response.json()

    def models(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/models", timeout=20)
        response.raise_for_status()
        return response.json()

    def tokenize(self, text: str) -> int:
        url = self.base_url.removesuffix("/v1") + "/tokenize"
        response = self.session.post(
            url,
            json={"content": text, "add_special": False, "parse_special": True},
            timeout=30,
        )
        response.raise_for_status()
        return len(response.json().get("tokens", []))

    def call(
        self,
        messages: List[Dict[str, str]],
        agent_name: str,
        seed: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        generation = self.agent_generation[agent_name]
        max_tokens = int(generation["max_tokens"])
        reasoning_effort = str(
            generation.get("reasoning_effort", "low")
        ).lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                f"Unsupported reasoning_effort for {agent_name}: "
                f"{reasoning_effort}"
            )

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(generation["temperature"]),
            "max_tokens": max_tokens,
            "seed": int(seed),
            "cache_prompt": self.cache_prompt,
            "reasoning_format": self.reasoning_format,
            "chat_template_kwargs": {
                "reasoning_effort": reasoning_effort
            },
            "timings_per_token": True,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Optional[Exception] = None
        retry_count = 0
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Retryable HTTP {response.status_code}: "
                        f"{response.text[:300]}",
                        response=response,
                    )
                response.raise_for_status()
                latency = time.perf_counter() - started
                data = response.json()
                choice = data["choices"][0]
                message = choice.get("message", {}) or {}
                raw_content = message.get("content") or ""
                reasoning = (
                    message.get("reasoning_content")
                    or message.get("thinking")
                    or ""
                )
                final_content, markers = extract_final_content(raw_content)
                finish_reason = choice.get("finish_reason")
                usage = data.get("usage", {}) or {}
                completion_tokens = int(
                    usage.get("completion_tokens", 0) or 0
                )
                truncated = (
                    str(finish_reason).lower() in {"length", "limit"}
                    or completion_tokens >= max_tokens
                )
                return LLMResult(
                    output=final_content,
                    usage=usage,
                    timings=data.get("timings", {}) or {},
                    finish_reason=finish_reason,
                    latency_sec=latency,
                    retry_count=retry_count,
                    seed=seed,
                    truncated=truncated,
                    empty_output=not bool(final_content.strip()),
                    raw_reasoning_markers=markers,
                    reasoning_chars=len(reasoning),
                    raw_content_chars=len(raw_content),
                )
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                retry_count += 1
                time.sleep(self.backoff * (2**attempt))

        raise RuntimeError(
            f"LLM request failed after {self.max_retries + 1} attempts"
        ) from last_error

    def warmup(self, calls: int) -> None:
        for index in range(calls):
            self.call(
                [
                    {
                        "role": "system",
                        "content": "Return only the word READY.",
                    },
                    {"role": "user", "content": "Warm-up request."},
                ],
                agent_name="Evaluator",
                seed=900000 + index,
            )
