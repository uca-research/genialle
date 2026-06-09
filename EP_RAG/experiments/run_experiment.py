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
from app.core.pedagogical_renderer import render_pedagogical_answer

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "experiments" / "config" / "scenarios.json"
QUESTIONS_PATH = BASE_DIR / "experiments" / "questions" / "questions_v1.jsonl"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

VALID_MODES = {
    "compare_contrast",
    "worked_example",
    "misconception_first",
}

VALID_FOCUS = {
    "definition",
    "memory_layout",
    "access_time",
    "insertion_deletion",
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
    pattern = r"(pág\.\s*\d+|page\s*\d+)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

def has_worked_example(text: str) -> bool:
    lower = text.lower()
    return "worked_example:" in lower or "example" in lower

def has_key_idea(text: str) -> bool:
    lower = text.lower()
    return "key_idea:" in lower or "key idea" in lower

def has_typical_error(text: str) -> bool:
    lower = text.lower()
    return "typical_error:" in lower or "typical error" in lower

def has_check_question(text: str) -> bool:
    lower = text.lower()
    return "check_question:" in lower or "check question" in lower

def distinct_retrieved_sources(retrieved: list) -> int:
    return len({item["metadata"]["source_file"] for item in retrieved})

def distinct_agent1_sources(agent1_payload: dict) -> int:
    ev = agent1_payload.get("retrieved_evidence", [])
    return len({item.get("source", "") for item in ev})

def agent1_parse_success(agent1_payload: dict) -> bool:
    gaps = agent1_payload.get("knowledge_gaps", [])
    return "Fallo de parseo JSON en agente 1" not in gaps

def clean_source_name(source: str) -> str:
    return re.sub(r",\s*pág\.\s*\d+\s*$", "", source, flags=re.IGNORECASE).strip()

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

def build_steps(mode: str, s1: str, s2: str, s3: str):
    if mode == "worked_example":
        return [
            f"A linked list is a sequence of nodes. Each node stores a value and a reference to the next node, so the elements do not need to occupy contiguous memory ({s3}).",
            f"An array stores its elements in contiguous memory. This makes direct access by position very fast, but it also depends on having a continuous memory block available ({s1}).",
            f"As a rule of thumb, arrays are convenient for direct indexed access, while linked lists can be useful when the structure is modified through links between nodes ({s3}).",
        ]

    if mode == "misconception_first":
        return [
            f"A common misconception is that a linked list is just a more flexible array. It is not: its internal structure is different because it is built from linked nodes ({s3}).",
            f"In an array, elements are stored in contiguous memory, which supports fast indexed access ({s1}). In a linked list, reaching a specific element usually requires traversing nodes one by one ({s3}).",
            f"In practice, arrays usually favor direct access, while linked lists may be more suitable when modifications are made through structural links between nodes ({s2}; {s3}).",
        ]

    return [
        f"A linked list is a structure made of nodes. Each node stores a value and a reference to the next node, so the sequence is traversed node by node ({s3}).",
        f"An array stores elements in contiguous memory. That layout allows fast direct access to a specific position because the location depends on the index ({s1}).",
        f"The main difference is a trade-off between direct access and structural flexibility: arrays usually favor indexed access, while linked lists represent sequences through references between nodes ({s1}; {s3}).",
    ]

def build_worked_example(s1: str, s3: str):
    return (
        "Imagine that you want to store the values 10, 20, and 30.\n"
        "- In an array, you can picture three contiguous memory slots: [10, 20, 30].\n"
        "- In a linked list, you can picture three connected nodes: 10 -> 20 -> 30.\n"
        "If you want to read the second element directly, the array is usually more convenient. "
        "If you want to insert a new node between two already located nodes, the linked list can be more natural from a structural point of view "
        f"({s1}; {s3})."
    )

def build_key_idea():
    return "The key idea is not that one structure is always better, but that arrays and linked lists solve different needs."

def build_typical_error():
    return "A typical mistake is to assume that a linked list always replaces an array or always saves memory. The right choice depends on the required access and modification pattern."

def render_pedagogical_answer_en(agent1_payload: dict, agent2_plan: dict) -> str:
    s1, s2, s3 = top_sources(agent1_payload)

    mode = agent2_plan.get("mode", "compare_contrast")
    check_question = agent2_plan.get(
        "check_question",
        "What is the main difference between a linked list and an array?"
    )
    rationale = agent2_plan.get(
        "rationale",
        "Comparing both structures helps reduce cognitive load for novice learners."
    )

    steps = build_steps(mode, s1, s2, s3)
    worked_example = build_worked_example(s1, s3)
    key_idea = build_key_idea()
    typical_error = build_typical_error()

    return (
        "student_answer:\n"
        f"1. {steps[0]}\n"
        f"2. {steps[1]}\n"
        f"3. {steps[2]}\n\n"
        "worked_example:\n"
        f"{worked_example}\n\n"
        "key_idea:\n"
        f"{key_idea}\n\n"
        "typical_error:\n"
        f"{typical_error}\n\n"
        "check_question:\n"
        f"{check_question}\n\n"
        "didactic_notes:\n"
        f"- pedagogical_mode: {mode}\n"
        f"- main_rationale: {rationale}"
    )

def scenario_a_answer(question: str, retrieved: list):
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

def build_granite_direct_sections_prompt(agent1_payload: dict) -> str:
    return f"""
You are a pedagogical instructor for introductory computer science.

Write the final student-facing answer directly in English.
Do not show your reasoning.
Do not add preambles.
Do not mention that you are planning.
Use only the structured evidence below.

Structured evidence:
{agent1_payload}

Return exactly this format:

student_answer:
1. ...
2. ...
3. ...

worked_example:
...

key_idea:
...

typical_error:
...

check_question:
...

didactic_notes:
- pedagogical_mode: ...
- main_rationale: ...
""".strip()

def direct_sections_parse_success(text: str) -> bool:
    lower = text.lower()
    return all([
        "student_answer:" in lower,
        "worked_example:" in lower,
        "key_idea:" in lower,
        "typical_error:" in lower,
        "check_question:" in lower,
        "didactic_notes:" in lower,
    ])

def build_granite_plan_prompt(agent1_payload: dict) -> str:
    return f"""
You are a pedagogical strategy selector for introductory computer science.

Do not explain the lesson.
Do not show your reasoning.
Do not add any extra text.

Use the structured evidence below.

Structured evidence:
{agent1_payload}

Return ONLY these 4 lines:

MODE: compare_contrast | worked_example | misconception_first
FOCUS: definition | memory_layout | access_time | insertion_deletion
CHECK_QUESTION: one short English question
RATIONALE: one short English sentence
""".strip()

def extract_field(text: str, field: str):
    pattern = rf"{field}\s*:\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def granite_planner(agent1_payload: dict):
    prompt = build_granite_plan_prompt(agent1_payload)
    raw_answer = generate_agent2_plain(
        prompt=prompt,
        max_new_tokens=120,
        temperature=0.1,
        do_sample=False
    )

    mode = extract_field(raw_answer, "MODE")
    focus = extract_field(raw_answer, "FOCUS")
    check_question = extract_field(raw_answer, "CHECK_QUESTION")
    rationale = extract_field(raw_answer, "RATIONALE")

    mode_ok = mode in VALID_MODES
    focus_ok = focus in VALID_FOCUS
    parse_success = all([mode_ok, focus_ok, check_question, rationale])

    plan = {
        "mode": mode if mode_ok else "compare_contrast",
        "focus": focus if focus_ok else "definition",
        "check_question": check_question or "What is the main difference between a linked list and an array?",
        "rationale": rationale or "Comparing both structures helps reduce cognitive load for novice learners.",
    }

    return plan, raw_answer, parse_success

def granite_direct_writer(agent1_payload: dict):
    prompt = build_granite_direct_sections_prompt(agent1_payload)
    answer = generate_agent2_plain(
        prompt=prompt,
        max_new_tokens=420,
        temperature=0.1,
        do_sample=False
    )
    parse_success = direct_sections_parse_success(answer)
    return answer, parse_success

def fixed_curated_plan():
    return {
        "mode": "compare_contrast",
        "focus": "definition",
        "check_question": "What is the main difference between a linked list and an array?",
        "rationale": "A compare-and-contrast strategy helps organize core differences for novice learners."
    }

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
    agent2_plan = None
    agent2_raw = None
    answer = ""
    agent1_ms = 0.0
    agent2_ms = 0.0
    agent2_parse_success = None

    if scenario_id == "scenario_a":
        started_agent1 = timer_start()
        answer = scenario_a_answer(question_obj["question"], retrieved)
        agent1_ms = timer_ms(started_agent1)

    elif scenario_id == "scenario_b":
        started_agent1 = timer_start()
        agent1_payload = run_agent1(question_obj["question"], "novice", retrieved)
        agent1_ms = timer_ms(started_agent1)

        agent2_plan = fixed_curated_plan()
        answer = render_pedagogical_answer_en(agent1_payload, agent2_plan)
        agent2_ms = 0.0
        agent2_parse_success = None

    elif scenario_id == "scenario_c":
        started_agent1 = timer_start()
        agent1_payload = run_agent1(question_obj["question"], "novice", retrieved)
        agent1_ms = timer_ms(started_agent1)

        started_agent2 = timer_start()
        agent2_plan, agent2_raw, agent2_parse_success = granite_planner(agent1_payload)
        agent2_ms = timer_ms(started_agent2)

        answer = render_pedagogical_answer_en(agent1_payload, agent2_plan)

    elif scenario_id == "scenario_d":
        started_agent1 = timer_start()
        agent1_payload = run_agent1(question_obj["question"], "novice", retrieved)
        agent1_ms = timer_ms(started_agent1)

        started_agent2 = timer_start()
        answer, agent2_parse_success = granite_direct_writer(agent1_payload)
        agent2_ms = timer_ms(started_agent2)

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
        "agent2_plan": agent2_plan,
        "agent2_raw": agent2_raw,
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
                "has_worked_example": has_worked_example(answer),
                "has_key_idea": has_key_idea(answer),
                "has_typical_error": has_typical_error(answer),
                "has_check_question": has_check_question(answer),
            },
            "grounding": {
                "agent1_parse_success": agent1_parse_success(agent1_payload) if agent1_payload else None,
                "agent2_parse_success": agent2_parse_success,
                "agent1_evidence_count": len(agent1_payload.get("retrieved_evidence", [])) if agent1_payload else 0,
                "agent1_distinct_sources": distinct_agent1_sources(agent1_payload) if agent1_payload else 0,
            },
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
            print(
                f"[done] {scenario_id} | rep={rep} | question={q['id']} | total_ms={record['metrics']['latency_ms']['total']}"
            )

    release_models()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all", help="scenario_a, scenario_b, scenario_c, scenario_d, or all")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions")
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetitions")
    parser.add_argument("--warmup", type=int, default=None, help="Override warm-up requests")
    args = parser.parse_args()

    scenarios = load_scenarios(CONFIG_PATH)
    questions = load_questions(QUESTIONS_PATH, limit=args.limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.scenario == "all":
        selected = ["scenario_a", "scenario_b", "scenario_c", "scenario_d"]
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
