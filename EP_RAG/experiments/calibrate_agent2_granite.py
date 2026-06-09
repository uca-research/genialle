import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.core.retriever import search
from app.agents.agent1 import run_agent1
from app.core.local_llm_runtime import (
    generate_agent2_plain,
    unload_qwen,
    unload_deepseek,
)
from app.core.telemetry import (
    timer_start,
    timer_ms,
    process_memory_mb,
    cuda_metrics,
    reset_cuda_peak_memory,
)
from app.core.pedagogical_renderer import render_pedagogical_answer

BASE_DIR = Path("/home/albertomatilla/rag_agentico_edu_exp2_granite")
QUESTIONS_PATH = BASE_DIR / "experiments" / "questions" / "questions_v1.jsonl"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

VALID_MODES = {"compare_contrast", "worked_example", "misconception_first"}
VALID_FOCUS = {"definition", "memory_layout", "access_time", "insertion_deletion"}

def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

def build_direct_sections_prompt(agent1_payload: dict) -> str:
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

def build_direct_json_prompt(agent1_payload: dict) -> str:
    return f"""
You are a pedagogical instructor for introductory computer science.

Write the final student-facing answer directly in English.
Do not show your reasoning.
Do not add preambles.

Use only the structured evidence below.

Structured evidence:
{agent1_payload}

Return ONLY valid JSON with this exact schema:

{{
  "student_answer_steps": ["...", "...", "..."],
  "worked_example": "...",
  "key_idea": "...",
  "typical_error": "...",
  "check_question": "...",
  "didactic_notes": {{
    "pedagogical_mode": "...",
    "main_rationale": "..."
  }}
}}
""".strip()

def build_plan_lines_prompt(agent1_payload: dict) -> str:
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

def build_plan_json_prompt(agent1_payload: dict) -> str:
    return f"""
You are a pedagogical strategy selector for introductory computer science.

Do not explain the lesson.
Do not show your reasoning.
Do not add any extra text.

Use the structured evidence below.

Structured evidence:
{agent1_payload}

Return ONLY valid JSON with this exact schema:

{{
  "mode": "compare_contrast or worked_example or misconception_first",
  "focus": "definition or memory_layout or access_time or insertion_deletion",
  "check_question": "...",
  "rationale": "..."
}}
""".strip()

def extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return None

def extract_field(text: str, field: str):
    pattern = rf"{field}\s*:\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def parse_direct_sections(text: str):
    lower = text.lower()
    return {
        "parse_success": all([
            "student_answer:" in lower,
            "worked_example:" in lower,
            "key_idea:" in lower,
            "typical_error:" in lower,
            "check_question:" in lower,
            "didactic_notes:" in lower,
        ]),
        "has_student_answer": "student_answer:" in lower,
        "has_worked_example": "worked_example:" in lower,
        "has_key_idea": "key_idea:" in lower,
        "has_typical_error": "typical_error:" in lower,
        "has_check_question": "check_question:" in lower,
        "has_didactic_notes": "didactic_notes:" in lower,
    }

def parse_direct_json(text: str):
    obj = extract_json_block(text)
    if obj is None:
        return {
            "parse_success": False,
            "parsed": None
        }

    ok = all([
        "student_answer_steps" in obj,
        "worked_example" in obj,
        "key_idea" in obj,
        "typical_error" in obj,
        "check_question" in obj,
        "didactic_notes" in obj,
    ])

    return {
        "parse_success": ok,
        "parsed": obj
    }

def parse_plan_lines(text: str):
    mode = extract_field(text, "MODE")
    focus = extract_field(text, "FOCUS")
    check_question = extract_field(text, "CHECK_QUESTION")
    rationale = extract_field(text, "RATIONALE")

    mode_ok = mode in VALID_MODES
    focus_ok = focus in VALID_FOCUS
    ok = all([mode_ok, focus_ok, check_question, rationale])

    return {
        "parse_success": ok,
        "parsed": {
            "mode": mode if mode_ok else None,
            "focus": focus if focus_ok else None,
            "check_question": check_question,
            "rationale": rationale
        }
    }

def parse_plan_json(text: str):
    obj = extract_json_block(text)
    if obj is None:
        return {
            "parse_success": False,
            "parsed": None
        }

    mode = obj.get("mode")
    focus = obj.get("focus")
    check_question = obj.get("check_question")
    rationale = obj.get("rationale")

    mode_ok = mode in VALID_MODES
    focus_ok = focus in VALID_FOCUS
    ok = all([mode_ok, focus_ok, check_question, rationale])

    return {
        "parse_success": ok,
        "parsed": {
            "mode": mode if mode_ok else None,
            "focus": focus if focus_ok else None,
            "check_question": check_question,
            "rationale": rationale
        }
    }

def normalize_plan(plan: dict):
    return {
        "mode": plan.get("mode", "compare_contrast"),
        "focus": plan.get("focus", "definition"),
        "check_question": plan.get("check_question", "What is the main difference between the two concepts?"),
        "rationale": plan.get("rationale", "A simple comparison reduces cognitive load for novice learners.")
    }

TEMPLATES = [
    {
        "template_id": "direct_sections",
        "mode": "without_rendering",
        "builder": build_direct_sections_prompt,
        "parser": parse_direct_sections,
        "max_new_tokens": 420,
    },
    {
        "template_id": "direct_json",
        "mode": "without_rendering",
        "builder": build_direct_json_prompt,
        "parser": parse_direct_json,
        "max_new_tokens": 420,
    },
    {
        "template_id": "plan_lines",
        "mode": "with_rendering",
        "builder": build_plan_lines_prompt,
        "parser": parse_plan_lines,
        "max_new_tokens": 120,
    },
    {
        "template_id": "plan_json",
        "mode": "with_rendering",
        "builder": build_plan_json_prompt,
        "parser": parse_plan_json,
        "max_new_tokens": 160,
    },
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    questions = load_questions(QUESTIONS_PATH, limit=args.limit)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"calibration_granite_{timestamp}.jsonl"

    print(f"[calibration] output={out_path}")
    print(f"[calibration] questions={len(questions)}")

    for i in range(min(args.warmup, len(questions))):
        q = questions[i]
        retrieved = search(q["question"], top_k=5)
        agent1_payload = run_agent1(q["question"], "novice", retrieved)
        prompt = build_plan_lines_prompt(agent1_payload)
        _ = generate_agent2_plain(prompt, max_new_tokens=120, temperature=args.temperature, do_sample=False)

    for q in questions:
        print(f"[question] {q['id']} | {q['question']}")
        retrieved = search(q["question"], top_k=5)
        agent1_payload = run_agent1(q["question"], "novice", retrieved)

        for tpl in TEMPLATES:
            reset_cuda_peak_memory()
            started = timer_start()

            prompt = tpl["builder"](agent1_payload)
            raw_output = generate_agent2_plain(
                prompt,
                max_new_tokens=tpl["max_new_tokens"],
                temperature=args.temperature,
                do_sample=False
            )

            latency_ms = timer_ms(started)
            parsed = tpl["parser"](raw_output)

            rendered_answer = None
            rendered_sections = None

            if tpl["mode"] == "with_rendering" and parsed["parse_success"]:
                plan = normalize_plan(parsed["parsed"])
                rendered_answer = render_pedagogical_answer(agent1_payload, plan)
                rendered_sections = {
                    "has_student_answer": "student_answer:" in rendered_answer.lower() or "respuesta_para_estudiante:" in rendered_answer.lower(),
                    "has_worked_example": "worked_example:" in rendered_answer.lower() or "ejemplo_trabajado:" in rendered_answer.lower(),
                    "has_key_idea": "key_idea:" in rendered_answer.lower() or "idea_clave:" in rendered_answer.lower(),
                    "has_typical_error": "typical_error:" in rendered_answer.lower() or "error_tipico:" in rendered_answer.lower(),
                    "has_check_question": "check_question:" in rendered_answer.lower() or "pregunta_de_comprobacion:" in rendered_answer.lower(),
                }

            record = {
                "question_id": q["id"],
                "question": q["question"],
                "template_id": tpl["template_id"],
                "mode": tpl["mode"],
                "latency_ms": latency_ms,
                "agent1_payload": agent1_payload,
                "raw_output": raw_output,
                "parse_success": parsed["parse_success"],
                "parsed_output": parsed.get("parsed"),
                "rendered_answer": rendered_answer,
                "rendered_sections": rendered_sections,
                "memory": {
                    "process_rss_mb": process_memory_mb(),
                    **cuda_metrics()
                }
            }

            append_jsonl(out_path, record)
            print(f"[done] {tpl['template_id']} | parse_success={parsed['parse_success']} | latency_ms={latency_ms}")

    unload_qwen()
    unload_deepseek()

if __name__ == "__main__":
    main()
