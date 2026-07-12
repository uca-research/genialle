from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from app_v2.agents import AdapterAgent, EvaluatorAgent, TeacherAgent
from app_v2.llm_client import LLMClient, LLMResult
from app_v2.memory.base import BaseMemory, MemoryRead
from app_v2.metrics.energy import NVMLPowerMonitor
from app_v2.output_parser import parse_json_object


AGENT_OFFSETS = {"Teacher": 1, "Adapter": 2, "Evaluator": 3}


def matched_seed(
    master_seed: int,
    task_index: int,
    repetition: int,
    round_idx: int,
    agent_name: str,
) -> int:
    # Memory condition is deliberately excluded.
    return (
        master_seed
        + repetition * 10000
        + task_index * 100
        + round_idx * 10
        + AGENT_OFFSETS[agent_name]
    )


def normalized_usage(result: LLMResult) -> Dict[str, int]:
    usage = result.usage or {}
    details = usage.get("prompt_tokens_details", {}) or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cached_tokens": int(details.get("cached_tokens", 0) or 0),
    }


class EpisodeRunner:
    def __init__(
        self,
        client: LLMClient,
        memory: BaseMemory,
        monitor: NVMLPowerMonitor,
        settings: Dict[str, Any],
    ):
        self.client = client
        self.memory = memory
        self.monitor = monitor
        self.settings = settings
        self.teacher = TeacherAgent(client)
        self.adapter = AdapterAgent(client)
        self.evaluator = EvaluatorAgent(client)

    def _execute_agent(
        self,
        agent: Any,
        task: Dict[str, Any],
        round_data: Dict[str, Any],
        memory_read: MemoryRead,
        seed: int,
        base: Dict[str, Any],
        upstream_output: Optional[str] = None,
    ) -> Tuple[LLMResult, Dict[str, Any], float, float]:
        start = time.perf_counter()
        result = agent.step(
            task=task,
            round_data=round_data,
            memory_context=memory_read.text,
            seed=seed,
            upstream_output=upstream_output,
        )
        end = time.perf_counter()
        step = {
            **base,
            "seed": seed,
            "call_start_monotonic": start,
            "call_end_monotonic": end,
            "latency_sec": result.latency_sec,
            "finish_reason": result.finish_reason,
            "retry_count": result.retry_count,
            "truncated": result.truncated,
            "empty_output": result.empty_output,
            "raw_reasoning_markers": result.raw_reasoning_markers,
            "reasoning_chars": result.reasoning_chars,
            "raw_content_chars": result.raw_content_chars,
            "output_chars": len(result.output),
            "output_sha256": hashlib.sha256(
                result.output.encode("utf-8")
            ).hexdigest(),
            "output": result.output,
            "memory_read": memory_read.metrics(),
            "usage": normalized_usage(result),
            "server_timings": result.timings,
        }
        return result, step, start, end

    def run(
        self,
        run_key: str,
        condition: str,
        task: Dict[str, Any],
        task_index: int,
        repetition: int,
    ) -> Dict[str, Any]:
        master_seed = int(self.settings["experiment"]["master_seed"])
        episode_id = f"{run_key}-{time.time_ns()}"
        self.memory.start_episode(episode_id, task["learner_profile"])

        baseline = self.monitor.measure_baseline(
            float(self.settings["energy"]["baseline_duration_sec"])
        )
        monitor_start = self.monitor.start()
        wall_start = time.perf_counter()

        steps: List[Dict[str, Any]] = []
        intervals: List[Tuple[int, float, float]] = []
        evaluator_scores: List[Dict[str, Any]] = []
        evaluator_parse_failures = 0

        try:
            step_idx = 0
            for round_idx, round_data in enumerate(task["rounds"]):
                query = (
                    task["learning_goal"]
                    + "\n"
                    + round_data["learner_event"]
                    + "\n"
                    + round_data["adaptation_target"]
                )

                read_start = time.perf_counter()
                teacher_memory = self.memory.get_context(
                    episode_id, "Teacher", query, round_idx
                )
                read_end = time.perf_counter()
                teacher_result, teacher_step, start, end = self._execute_agent(
                    self.teacher,
                    task,
                    round_data,
                    teacher_memory,
                    matched_seed(
                        master_seed,
                        task_index,
                        repetition,
                        round_idx,
                        "Teacher",
                    ),
                    {
                        "run_key": run_key,
                        "episode_id": episode_id,
                        "condition": condition,
                        "task_id": task["id"],
                        "repetition": repetition,
                        "round_idx": round_idx,
                        "step_idx": step_idx,
                        "agent": "Teacher",
                        "memory_read_sec": read_end - read_start,
                    },
                )
                self.memory.update(
                    episode_id,
                    "Teacher",
                    round_idx,
                    teacher_result.output,
                    {"task_id": task["id"]},
                )
                steps.append(teacher_step)
                intervals.append((len(steps) - 1, start, end))
                step_idx += 1

                read_start = time.perf_counter()
                adapter_memory = self.memory.get_context(
                    episode_id, "Adapter", query, round_idx
                )
                read_end = time.perf_counter()
                adapter_result, adapter_step, start, end = self._execute_agent(
                    self.adapter,
                    task,
                    round_data,
                    adapter_memory,
                    matched_seed(
                        master_seed,
                        task_index,
                        repetition,
                        round_idx,
                        "Adapter",
                    ),
                    {
                        "run_key": run_key,
                        "episode_id": episode_id,
                        "condition": condition,
                        "task_id": task["id"],
                        "repetition": repetition,
                        "round_idx": round_idx,
                        "step_idx": step_idx,
                        "agent": "Adapter",
                        "memory_read_sec": read_end - read_start,
                    },
                    upstream_output=teacher_result.output,
                )
                self.memory.update(
                    episode_id,
                    "Adapter",
                    round_idx,
                    adapter_result.output,
                    {"task_id": task["id"]},
                )
                steps.append(adapter_step)
                intervals.append((len(steps) - 1, start, end))
                step_idx += 1

                evaluator_result, evaluator_step, start, end = self._execute_agent(
                    self.evaluator,
                    task,
                    round_data,
                    MemoryRead(visibility="evaluator_isolated"),
                    matched_seed(
                        master_seed,
                        task_index,
                        repetition,
                        round_idx,
                        "Evaluator",
                    ),
                    {
                        "run_key": run_key,
                        "episode_id": episode_id,
                        "condition": condition,
                        "task_id": task["id"],
                        "repetition": repetition,
                        "round_idx": round_idx,
                        "step_idx": step_idx,
                        "agent": "Evaluator",
                        "memory_read_sec": 0.0,
                    },
                    upstream_output=adapter_result.output,
                )
                try:
                    rubric = parse_json_object(evaluator_result.output)
                    evaluator_step["evaluator_json_valid"] = True
                    evaluator_step["rubric"] = rubric
                    evaluator_scores.append(rubric)
                except Exception as exc:
                    evaluator_parse_failures += 1
                    evaluator_step["evaluator_json_valid"] = False
                    evaluator_step["rubric"] = None
                    evaluator_step["evaluator_parse_error"] = repr(exc)
                steps.append(evaluator_step)
                intervals.append((len(steps) - 1, start, end))
                step_idx += 1

                # The learner event becomes longitudinal state only after the
                # complete current round, so no condition can read it early.
                self.memory.record_round_state(
                    episode_id, round_idx, round_data
                )

            wall_end = time.perf_counter()
            power = self.monitor.stop()
            for index, start, end in intervals:
                steps[index]["gpu_energy"] = self.monitor.interval_energy(
                    start, end
                )

            usage = {
                key: sum(int(step["usage"][key]) for step in steps)
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cached_tokens",
                )
            }

            snapshot = self.memory.snapshot(episode_id)
            deletion_success = self.memory.clear_episode(episode_id)

            episode = {
                "status": "completed",
                "run_key": run_key,
                "episode_id": episode_id,
                "condition": condition,
                "task_id": task["id"],
                "task_domain": task["domain"],
                "task_level": task["level"],
                "task_index": task_index,
                "repetition": repetition,
                "rounds": len(task["rounds"]),
                "llm_calls": len(steps),
                "wall_time_sec": wall_end - wall_start,
                "monitor_start_monotonic": monitor_start,
                "baseline": baseline,
                "power": power,
                "usage": usage,
                "robustness": {
                    "truncated_calls": sum(
                        bool(step["truncated"]) for step in steps
                    ),
                    "empty_outputs": sum(
                        bool(step["empty_output"]) for step in steps
                    ),
                    "raw_reasoning_marker_outputs": sum(
                        bool(step["raw_reasoning_markers"]) for step in steps
                    ),
                    "request_retries": sum(
                        int(step["retry_count"]) for step in steps
                    ),
                    "evaluator_parse_failures": evaluator_parse_failures,
                },
                "memory": {
                    "context_chars_read_total": sum(
                        int(step["memory_read"]["context_chars"])
                        for step in steps
                    ),
                    "fragments_read_total": sum(
                        int(step["memory_read"]["fragment_count"])
                        for step in steps
                    ),
                    "cross_agent_fragments_read_total": sum(
                        int(step["memory_read"]["cross_agent_fragments"])
                        for step in steps
                    ),
                    "contains_current_round_reads": sum(
                        bool(step["memory_read"]["contains_current_round"])
                        for step in steps
                    ),
                    "snapshot_before_clear": snapshot,
                    "deletion_success": deletion_success,
                },
                "internal_evaluator_scores": evaluator_scores,
            }
            return {
                "episode": episode,
                "steps": steps,
                "power_samples": self.monitor.serialized_samples(),
            }
        except Exception:
            try:
                if self.monitor._thread is not None:
                    self.monitor.stop()
            except Exception:
                pass
            self.memory.clear_episode(episode_id)
            raise
