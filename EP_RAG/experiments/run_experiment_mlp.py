import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.core.retriever import search
from app.core.telemetry import (
    new_request_id,
    now_iso,
    process_memory_mb,
    cuda_metrics,
    reset_cuda_peak_memory,
    timer_start,
    timer_ms,
)
from app.core.local_llm_runtime import (
    generate_qwen_chat,
    generate_agent2_plain,
    unload_qwen,
    unload_deepseek,
)
from app.agents.agent1 import run_agent1
from app.core.min_learning_path import build_minimum_learning_path

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "experiments" / "config" / "scenarios_mlp.json"
QUESTIONS_PATH = BASE_DIR / "experiments" / "questions" / "questions_v1.jsonl"
RESULTS_DIR = BASE_DIR / "experiments" / "results_mlp"

load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

VALID_MODES = {
    "compare_contrast",
    "worked_example",
    "misconception_first",
}

def load_scenarios(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}

def load_questions(path: Path, limit: int | None = None):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit is not None:
        questions = questions[:limit]
    return questions

def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def word_count(text: str) -> int:
    return len(text.split()) if text else 0

def count_page_citations(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"(page\s+\d+|pág\.\s*\d+)", text, flags=re.IGNORECASE))

def has_section(text: str, section_name: str) -> bool:
    return section_name.lower() in (text or "").lower()

def distinct_retrieved_sources(retrieved: list) -> int:
    return len({item["metadata"]["source_file"] for item in retrieved})

def distinct_agent1_sources(agent1_payload: dict) -> int:
    ev = agent1_payload.get("retrieved_evidence", [])
    return len({item.get("source", "") for item in ev})

def agent1_parse_success(agent1_payload: dict) -> bool:
    gaps = agent1_payload.get("knowledge_gaps", [])
    return "Fallo de parseo JSON en agente 1" not in gaps

def clean_source_name(source: str) -> str:
    return re.sub(r",\s*pág\.\s*\d+\s*$", "", source or "", flags=re.IGNORECASE).strip()

def render_source(ev: dict) -> str:
    source = clean_source_name(ev.get("source", "Unknown source"))
    page = ev.get("page", "?")
    return f"{source}, page {page}"

def top_sources(agent1_payload: dict):
    ev = agent1_payload.get("retrieved_evidence", [])
    if not ev:
        return ("Unknown source", "Unknown source", "Unknown source")
    s1 = render_source(ev[0]) if len(ev) > 0 else "Unknown source"
    s2 = render_source(ev[1]) if len(ev) > 1 else s1
    s3 = render_source(ev[2]) if len(ev) > 2 else s2
    return s1, s2, s3

def scenario_baseline_answer(question: str, retrieved: list):
    evidence_text = "\n\n".join(
        [
            f"[Source: {item['metadata']['source_file']}, page {item['metadata']['page']}, score={item['score']:.4f}]\n{item['text']}"
            for item in retrieved
        ]
    )

    prompt = f"""
You are a grounded computer science tutor.

Answer the question using only the evidence below.
Do not invent information outside the evidence.
Write in English.
Keep the answer concise but clear.
When possible, mention the source and page.

Question:
{question}

Evidence:
{evidence_text}
""".strip()

    return generate_qwen_chat(
        messages=[
            {"role": "system", "content": "You are a rigorous tutor who answers only from evidence."},
            {"role": "user", "content": prompt},
        ],
        max_new_tokens=350,
        temperature=0.1,
    )

def build_granite_curated_prompt(agent1_payload: dict) -> str:
    return f"""
You are a pedagogical synthesizer for introductory computer science.

Use the curated evidence below to generate the final instructional response.
Do not show hidden reasoning.
Do not add preambles.
Write in English.

Curated evidence:
{json.dumps(agent1_payload, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this exact schema:

{{
  "pedagogical_mode": "compare_contrast or worked_example or misconception_first",
  "student_answer_steps": ["...", "...", "..."],
  "worked_example": "...",
  "key_idea": "...",
  "typical_error": "...",
  "check_question": "...",
  "didactic_notes": {{
    "main_rationale": "..."
  }}
}}
""".strip()

def build_granite_mlp_prompt(agent1_payload: dict, mlp: dict) -> str:
    return f"""
You are a pedagogical synthesizer for introductory computer science.

Use the curated evidence and the minimum learning path below to generate the final instructional response.
Respect the minimum learning path by sequencing the explanation from prerequisite understanding to the target concept.
Do not show hidden reasoning.
Do not add preambles.
Write in English.

Curated evidence:
{json.dumps(agent1_payload, ensure_ascii=False, indent=2)}

Minimum learning path:
{json.dumps(mlp, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this exact schema:

{{
  "pedagogical_mode": "compare_contrast or worked_example or misconception_first",
  "minimum_learning_path": ["...", "...", "..."],
  "student_answer_steps": ["...", "...", "..."],
  "worked_example": "...",
  "key_idea": "...",
  "typical_error": "...",
  "check_question": "...",
  "didactic_notes": {{
    "main_rationale": "..."
  }}
}}
""".strip()

def extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None

def parse_pedagogical_json(text: str, require_mlp: bool = False):
    obj = extract_json_block(text)
    if obj is None:
        return None, False

    mode = obj.get("pedagogical_mode")
    steps = obj.get("student_answer_steps")
    worked_example = obj.get("worked_example")
    key_idea = obj.get("key_idea")
    typical_error = obj.get("typical_error")
    check_question = obj.get("check_question")
    didactic_notes = obj.get("didactic_notes")

    ok = all([
        mode in VALID_MODES,
        isinstance(steps, list) and len(steps) >= 2,
        isinstance(worked_example, str) and worked_example.strip(),
        isinstance(key_idea, str) and key_idea.strip(),
        isinstance(typical_error, str) and typical_error.strip(),
        isinstance(check_question, str) and check_question.strip(),
        isinstance(didactic_notes, dict) and isinstance(didactic_notes.get("main_rationale"), str) and didactic_notes.get("main_rationale").strip(),
    ])

    if require_mlp:
        mlp = obj.get("minimum_learning_path")
        ok = ok and isinstance(mlp, list) and len(mlp) >= 1

    return obj, ok

def render_pedagogical_json(agent1_payload: dict, pedagogical_obj: dict, include_mlp: bool = False) -> str:
    s1, s2, s3 = top_sources(agent1_payload)

    steps = pedagogical_obj.get("student_answer_steps", [])
    worked_example = pedagogical_obj.get("worked_example", "")
    key_idea = pedagogical_obj.get("key_idea", "")
    typical_error = pedagogical_obj.get("typical_error", "")
    check_question = pedagogical_obj.get("check_question", "")
    main_rationale = pedagogical_obj.get("didactic_notes", {}).get("main_rationale", "")
    mode = pedagogical_obj.get("pedagogical_mode", "compare_contrast")
    mlp = pedagogical_obj.get("minimum_learning_path", [])

    step_lines = []
    for idx, step in enumerate(steps[:3], start=1):
        source = s1 if idx == 1 else s2 if idx == 2 else s3
        step_lines.append(f"{idx}. {step} ({source}).")

    text = []
    if include_mlp and mlp:
        text.append("minimum_learning_path:")
        for idx, item in enumerate(mlp[:3], start=1):
            text.append(f"{idx}. {item}")
        text.append("")

    text.append("student_answer:")
    text.extend(step_lines)
    text.append("")
    text.append("worked_example:")
    text.append(f"{worked_example} ({s1}; {s3}).")
    text.append("")
    text.append("key_idea:")
    text.append(key_idea)
    text.append("")
    text.append("typical_error:")
    text.append(typical_error)
    text.append("")
    text.append("check_question:")
    text.append(check_question)
    text.append("")
    text.append("didactic_notes:")
    text.append(f"- pedagogical_mode: {mode}")
    text.append(f"- main_rationale: {main_rationale}")

    return "\n".join(text)

def memory_snapshot():
    return {
        "process_rss_mb": process_memory_mb(),
        **cuda_metrics()
    }

def run_one_request(scenario_id: str, question_obj: dict, top_k: int):
    request_id = new_request_id()
    reset_cuda_peak_memory()

    started_total = timer_start()

    started_retrieval = timer_start()
    retrieved = search(question_obj["question"], top_k=top_k)
    retrieval_ms = timer_ms(started_retrieval)

    agent1_payload = None
    agent2_raw = None
    pedagogical_obj = None
    mlp = None
    answer = ""
    agent1_ms = 0.0
    agent2_ms = 0.0
    agent2_parse_success = None

    if scenario_id == "baseline":
        started_agent1 = timer_start()
        answer = scenario_baseline_answer(question_obj["question"], retrieved)
        agent1_ms = timer_ms(started_agent1)

    elif scenario_id == "curated_pedagogical":
        started_agent1 = timer_start()
        agent1_payload = run_agent1(question_obj["question"], "novice", retrieved)
        agent1_ms = timer_ms(started_agent1)

        started_agent2 = timer_start()
        prompt = build_granite_curated_prompt(agent1_payload)
        agent2_raw = generate_agent2_plain(
            prompt,
            max_new_tokens=420,
            temperature=0.1,
            do_sample=False
        )
        pedagogical_obj, agent2_parse_success = parse_pedagogical_json(agent2_raw, require_mlp=False)
        agent2_ms = timer_ms(started_agent2)

        if pedagogical_obj is not None:
            answer = render_pedagogical_json(agent1_payload, pedagogical_obj, include_mlp=False)
        else:
            answer = "student_answer:\n1. No valid pedagogical JSON was returned."

    elif scenario_id == "curated_pedagogical_mlp":
        started_agent1 = timer_start()
        agent1_payload = run_agent1(question_obj["question"], "novice", retrieved)
        agent1_ms = timer_ms(started_agent1)

        mlp = build_minimum_learning_path(agent1_payload)

        started_agent2 = timer_start()
        prompt = build_granite_mlp_prompt(agent1_payload, mlp)
        agent2_raw = generate_agent2_plain(
            prompt,
            max_new_tokens=480,
            temperature=0.1,
            do_sample=False
        )
        pedagogical_obj, agent2_parse_success = parse_pedagogical_json(agent2_raw, require_mlp=True)
        agent2_ms = timer_ms(started_agent2)

        if pedagogical_obj is not None:
            answer = render_pedagogical_json(agent1_payload, pedagogical_obj, include_mlp=True)
        else:
            answer = "student_answer:\n1. No valid pedagogical JSON with minimum learning path was returned."

    else:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")

    total_ms = timer_ms(started_total)

    record = {
        "timestamp_utc": now_iso(),
        "request_id": request_id,
        "scenario_id": scenario_id,
        "question_id": question_obj["id"],
        "domain": question_obj["domain"],
        "question_type": question_obj["type"],
        "question": question_obj["question"],
        "top_k": top_k,
        "retrieval_count": len(retrieved),
        "retrieval_distinct_sources": distinct_retrieved_sources(retrieved),
        "agent1_payload": agent1_payload,
        "minimum_learning_path": mlp,
        "agent2_raw": agent2_raw,
        "pedagogical_obj": pedagogical_obj,
        "metrics": {
            "latency_ms": {
                "retrieval": retrieval_ms,
                "agent1": agent1_ms,
                "agent2": agent2_ms,
                "total": total_ms,
            },
            "memory": memory_snapshot(),
            "output": {
                "answer_chars": len(answer),
                "answer_words": word_count(answer),
                "page_citation_count": count_page_citations(answer),
                "has_worked_example": has_section(answer, "worked_example:"),
                "has_key_idea": has_section(answer, "key_idea:"),
                "has_typical_error": has_section(answer, "typical_error:"),
                "has_check_question": has_section(answer, "check_question:"),
                "has_mlp_section": has_section(answer, "minimum_learning_path:"),
            },
            "grounding": {
                "agent1_parse_success": agent1_parse_success(agent1_payload) if agent1_payload else None,
                "agent2_parse_success": agent2_parse_success,
                "agent1_evidence_count": len(agent1_payload.get("retrieved_evidence", [])) if agent1_payload else 0,
                "agent1_distinct_sources": distinct_agent1_sources(agent1_payload) if agent1_payload else 0,
            },
            "mlp": {
                "path_length": mlp.get("path_length", 0) if mlp else 0,
            }
        },
        "answer": answer,
    }

    return record

def warmup_scenario(scenario_id: str, questions: list, warmup_requests: int, top_k: int):
    if warmup_requests <= 0:
        return
    print(f"[warmup] Starting {warmup_requests} warm-up requests for {scenario_id}")
    for i in range(min(warmup_requests, len(questions))):
        _ = run_one_request(scenario_id, questions[i], top_k=top_k)
    print(f"[warmup] Completed for {scenario_id}")

def release_models():
    unload_qwen()
    unload_deepseek()

def run_scenario(scenario: dict, questions: list, results_path: Path, repetitions_override: int | None = None, warmup_override: int | None = None):
    scenario_id = scenario["id"]
    top_k = scenario["top_k"]
    repetitions = repetitions_override if repetitions_override is not None else scenario["repetitions"]
    warmup_requests = warmup_override if warmup_override is not None else scenario["warmup_requests"]

    print(f"[scenario] {scenario_id}")
    print(f"[scenario] top_k={top_k}, warmup_requests={warmup_requests}, repetitions={repetitions}")
    print(f"[scenario] results_path={results_path}")

    warmup_scenario(scenario_id, questions, warmup_requests, top_k)

    counter = 0
    for rep in range(1, repetitions + 1):
        print(f"[scenario] repetition {rep}/{repetitions}")
        for q in questions:
            counter += 1
            record = run_one_request(scenario_id, q, top_k)
            record["repetition"] = rep
            record["sequence_id"] = counter
            append_jsonl(results_path, record)
            print(f"[done] {scenario_id} | rep={rep} | question={q['id']} | total_ms={record['metrics']['latency_ms']['total']}")

    release_models()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all", help="baseline, curated_pedagogical, curated_pedagogical_mlp, or all")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions")
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetitions")
    parser.add_argument("--warmup", type=int, default=None, help="Override warm-up requests")
    args = parser.parse_args()

    scenarios = load_scenarios(CONFIG_PATH)
    questions = load_questions(QUESTIONS_PATH, limit=args.limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.scenario == "all":
        selected = ["baseline", "curated_pedagogical", "curated_pedagogical_mlp"]
    else:
        if args.scenario not in scenarios:
            raise ValueError(f"Unknown scenario: {args.scenario}")
        selected = [args.scenario]

    for scenario_id in selected:
        scenario = scenarios[scenario_id]
        results_path = RESULTS_DIR / f"{scenario_id}_{timestamp}.jsonl"
        run_scenario(
            scenario=scenario,
            questions=questions,
            results_path=results_path,
            repetitions_override=args.repetitions,
            warmup_override=args.warmup,
        )

if __name__ == "__main__":
    main()
