import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "experiments" / "results_mlp"

def p95(values):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round(0.95 * (len(values) - 1)))
    return values[idx]

def load_records():
    records = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["_file"] = path.name
                    records.append(record)
    return records

def summarize_group(records):
    retrieval = [r["metrics"]["latency_ms"]["retrieval"] for r in records]
    agent1 = [r["metrics"]["latency_ms"]["agent1"] for r in records]
    agent2 = [r["metrics"]["latency_ms"]["agent2"] for r in records]
    total = [r["metrics"]["latency_ms"]["total"] for r in records]

    rss = [r["metrics"]["memory"]["process_rss_mb"] for r in records]
    cuda_reserved = [r["metrics"]["memory"]["cuda_reserved_mb"] for r in records]
    cuda_peak_reserved = [r["metrics"]["memory"]["cuda_max_reserved_mb"] for r in records]

    answer_chars = [r["metrics"]["output"]["answer_chars"] for r in records]
    answer_words = [r["metrics"]["output"]["answer_words"] for r in records]
    page_citations = [r["metrics"]["output"]["page_citation_count"] for r in records]

    has_example = [1 if r["metrics"]["output"]["has_worked_example"] else 0 for r in records]
    has_key_idea = [1 if r["metrics"]["output"]["has_key_idea"] else 0 for r in records]
    has_typical_error = [1 if r["metrics"]["output"]["has_typical_error"] else 0 for r in records]
    has_check_question = [1 if r["metrics"]["output"]["has_check_question"] else 0 for r in records]

    retrieval_sources = [r["retrieval_distinct_sources"] for r in records]
    agent1_evidence = [r["metrics"]["grounding"]["agent1_evidence_count"] for r in records]
    agent1_distinct_sources = [r["metrics"]["grounding"]["agent1_distinct_sources"] for r in records]

    agent1_parse_success = [
        1 if r["metrics"]["grounding"]["agent1_parse_success"] else 0
        for r in records
        if r["metrics"]["grounding"]["agent1_parse_success"] is not None
    ]
    agent2_parse_success = [
        1 if r["metrics"]["grounding"]["agent2_parse_success"] else 0
        for r in records
        if r["metrics"]["grounding"]["agent2_parse_success"] is not None
    ]

    return {
        "n": len(records),
        "retrieval_ms_mean": mean(retrieval),
        "retrieval_ms_median": median(retrieval),
        "retrieval_ms_p95": p95(retrieval),
        "agent1_ms_mean": mean(agent1),
        "agent1_ms_median": median(agent1),
        "agent1_ms_p95": p95(agent1),
        "agent2_ms_mean": mean(agent2),
        "agent2_ms_median": median(agent2),
        "agent2_ms_p95": p95(agent2),
        "total_ms_mean": mean(total),
        "total_ms_median": median(total),
        "total_ms_p95": p95(total),
        "rss_mb_mean": mean(rss),
        "cuda_reserved_mb_mean": mean(cuda_reserved),
        "cuda_peak_reserved_mb_mean": mean(cuda_peak_reserved),
        "answer_chars_mean": mean(answer_chars),
        "answer_words_mean": mean(answer_words),
        "page_citations_mean": mean(page_citations),
        "has_example_rate": mean(has_example),
        "has_key_idea_rate": mean(has_key_idea),
        "has_typical_error_rate": mean(has_typical_error),
        "has_check_question_rate": mean(has_check_question),
        "retrieval_distinct_sources_mean": mean(retrieval_sources),
        "agent1_evidence_count_mean": mean(agent1_evidence),
        "agent1_distinct_sources_mean": mean(agent1_distinct_sources),
        "agent1_parse_success_rate": mean(agent1_parse_success) if agent1_parse_success else None,
        "agent2_parse_success_rate": mean(agent2_parse_success) if agent2_parse_success else None,
    }

def main():
    records = load_records()
    if not records:
        print("No experiment result files were found.")
        return

    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario_id"]].append(record)

    print(f"Loaded records: {len(records)}\n")

    for scenario_id, scenario_records in sorted(grouped.items()):
        summary = summarize_group(scenario_records)

        print(f"Scenario: {scenario_id}")
        print(f"  Requests: {summary['n']}")
        print(f"  Retrieval latency mean (ms): {summary['retrieval_ms_mean']:.2f}")
        print(f"  Retrieval latency median (ms): {summary['retrieval_ms_median']:.2f}")
        print(f"  Retrieval latency p95 (ms): {summary['retrieval_ms_p95']:.2f}")
        print(f"  Agent 1 latency mean (ms): {summary['agent1_ms_mean']:.2f}")
        print(f"  Agent 1 latency median (ms): {summary['agent1_ms_median']:.2f}")
        print(f"  Agent 1 latency p95 (ms): {summary['agent1_ms_p95']:.2f}")
        print(f"  Agent 2 latency mean (ms): {summary['agent2_ms_mean']:.2f}")
        print(f"  Agent 2 latency median (ms): {summary['agent2_ms_median']:.2f}")
        print(f"  Agent 2 latency p95 (ms): {summary['agent2_ms_p95']:.2f}")
        print(f"  Total latency mean (ms): {summary['total_ms_mean']:.2f}")
        print(f"  Total latency median (ms): {summary['total_ms_median']:.2f}")
        print(f"  Total latency p95 (ms): {summary['total_ms_p95']:.2f}")
        print(f"  RSS mean (MB): {summary['rss_mb_mean']:.2f}")
        print(f"  CUDA reserved mean (MB): {summary['cuda_reserved_mb_mean']:.2f}")
        print(f"  CUDA peak reserved mean (MB): {summary['cuda_peak_reserved_mb_mean']:.2f}")
        print(f"  Answer length mean (chars): {summary['answer_chars_mean']:.2f}")
        print(f"  Answer length mean (words): {summary['answer_words_mean']:.2f}")
        print(f"  Page citation mean: {summary['page_citations_mean']:.2f}")
        print(f"  Worked example rate: {summary['has_example_rate']:.2f}")
        print(f"  Key idea rate: {summary['has_key_idea_rate']:.2f}")
        print(f"  Typical error rate: {summary['has_typical_error_rate']:.2f}")
        print(f"  Check question rate: {summary['has_check_question_rate']:.2f}")
        print(f"  Retrieval distinct sources mean: {summary['retrieval_distinct_sources_mean']:.2f}")
        print(f"  Agent 1 evidence count mean: {summary['agent1_evidence_count_mean']:.2f}")
        print(f"  Agent 1 distinct sources mean: {summary['agent1_distinct_sources_mean']:.2f}")
        print(f"  Agent 1 parse success rate: {summary['agent1_parse_success_rate']}")
        print(f"  Agent 2 parse success rate: {summary['agent2_parse_success_rate']}")
        print()

if __name__ == "__main__":
    main()
