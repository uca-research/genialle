import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

RESULTS_DIR = Path("/home/albertomatilla/rag_agentico_edu_exp2_granite/experiments/results")

def load_records():
    files = sorted(RESULTS_DIR.glob("calibration_granite_*.jsonl"))
    if not files:
        return []
    latest = files[-1]
    print(f"Using file: {latest.name}")
    records = []
    with latest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    records = load_records()
    if not records:
        print("No calibration files found.")
        return

    groups = defaultdict(list)
    for r in records:
        groups[(r["mode"], r["template_id"])].append(r)

    for (mode, template_id), rows in sorted(groups.items()):
        lat = [r["latency_ms"] for r in rows]
        parse = [1 if r["parse_success"] else 0 for r in rows]
        rss = [r["memory"]["process_rss_mb"] for r in rows]
        cuda = [r["memory"]["cuda_max_reserved_mb"] for r in rows]

        print(f"\nMode: {mode} | Template: {template_id}")
        print(f"  n = {len(rows)}")
        print(f"  parse_success_rate = {mean(parse):.2f}")
        print(f"  latency_mean_ms = {mean(lat):.2f}")
        print(f"  latency_median_ms = {median(lat):.2f}")
        print(f"  rss_mean_mb = {mean(rss):.2f}")
        print(f"  cuda_peak_reserved_mean_mb = {mean(cuda):.2f}")

        if mode == "without_rendering":
            section_keys = [
                "has_student_answer",
                "has_worked_example",
                "has_key_idea",
                "has_typical_error",
                "has_check_question",
                "has_didactic_notes",
            ]
            for key in section_keys:
                vals = []
                for r in rows:
                    parsed = r.get("parsed_output")
                    if isinstance(parsed, dict) and key in parsed:
                        vals.append(1 if parsed[key] else 0)
                if vals:
                    print(f"  {key}_rate = {mean(vals):.2f}")

        if mode == "with_rendering":
            section_keys = [
                "has_student_answer",
                "has_worked_example",
                "has_key_idea",
                "has_typical_error",
                "has_check_question",
            ]
            for key in section_keys:
                vals = []
                for r in rows:
                    rendered = r.get("rendered_sections")
                    if isinstance(rendered, dict) and key in rendered:
                        vals.append(1 if rendered[key] else 0)
                if vals:
                    print(f"  rendered_{key}_rate = {mean(vals):.2f}")

if __name__ == "__main__":
    main()
