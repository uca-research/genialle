import json
import time

from app_v2.config import load_settings
from app_v2.llm_client import LLMClient
from app_v2.metrics.energy import NVMLPowerMonitor
from app_v2.tasks_loader import load_tasks


def main() -> None:
    settings = load_settings()
    tasks = load_tasks()
    client = LLMClient(settings)

    print("1. Server health")
    print(json.dumps(client.health(), indent=2))

    print("\n2. Model metadata")
    print(json.dumps(client.models(), indent=2)[:3000])

    print("\n3. Final-channel and seed test")
    result = client.call(
        [
            {
                "role": "system",
                "content": "Return only this exact text: PREFLIGHT_OK",
            },
            {"role": "user", "content": "Run preflight."},
        ],
        agent_name="Evaluator",
        seed=123456,
    )
    print(json.dumps({
        "output": result.output,
        "finish_reason": result.finish_reason,
        "truncated": result.truncated,
        "raw_reasoning_markers": result.raw_reasoning_markers,
        "reasoning_chars": result.reasoning_chars,
        "usage": result.usage,
        "timings": result.timings,
    }, indent=2))
    if not result.output:
        raise RuntimeError("Empty final content")
    if result.raw_reasoning_markers:
        raise RuntimeError(
            "Raw reasoning markers remain. Restart llama-server with "
            "--jinja --reasoning-format deepseek."
        )
    if result.truncated:
        raise RuntimeError("Preflight request was truncated")

    print("\n4. Tokenizer endpoint")
    tokens = client.tokenize("A short tokenizer preflight sentence.")
    print(f"Tokens: {tokens}")
    if tokens <= 0:
        raise RuntimeError("Tokenizer endpoint returned no tokens")

    print("\n5. NVML repeated sampling")
    monitor = NVMLPowerMonitor(
        gpu_index=int(settings["energy"]["gpu_index"]),
        sample_interval_sec=float(
            settings["energy"]["sample_interval_sec"]
        ),
    )
    try:
        print(json.dumps(monitor.device_info(), indent=2))
        baseline = monitor.measure_baseline(1.0)
        print(json.dumps(baseline, indent=2))
        monitor.start()
        time.sleep(1.0)
        summary = monitor.stop()
        print(json.dumps(summary, indent=2))
        if summary["sample_count"] < 2:
            raise RuntimeError("Insufficient power samples")
        if baseline["baseline_w_median"] <= 0:
            raise RuntimeError("Invalid idle baseline")
    finally:
        monitor.close()

    print("\n6. Task validation")
    print(f"Tasks: {len(tasks)}")
    print(f"Pilot episodes: {len(tasks) * 5 * settings['experiment']['pilot_repetitions']}")
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
