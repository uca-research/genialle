import re

from app.core.local_llm_runtime import generate_deepseek_user_prompt

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

def build_agent2_prompt(agent1_payload: dict) -> str:
    return f"""
Eres un selector de estrategia pedagógica para enseñar informática.

Tu tarea NO es explicar el contenido al estudiante.
Tu tarea NO es pensar en voz alta.
Tu tarea NO es redactar una respuesta larga.

Tu única tarea es elegir una estrategia didáctica adecuada para un estudiante novato.

Responde SOLO con estas 4 líneas y nada más:

MODE: compare_contrast | worked_example | misconception_first
FOCUS: definition | memory_layout | access_time | insertion_deletion
CHECK_QUESTION: una pregunta breve en español
RATIONALE: una frase breve en español

Entrada:
{agent1_payload}
""".strip()

def _extract_field(text: str, field: str) -> str | None:
    pattern = rf"{field}\s*:\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def run_agent2(agent1_payload: dict) -> dict:
    prompt = build_agent2_prompt(agent1_payload)

    raw_answer = generate_deepseek_user_prompt(
        prompt=prompt,
        max_new_tokens=220,
        temperature=0.2
    )

    print("[agent2][raw_answer]", repr(raw_answer[:800]))

    mode = _extract_field(raw_answer, "MODE") or "compare_contrast"
    focus = _extract_field(raw_answer, "FOCUS") or "definition"
    check_question = _extract_field(raw_answer, "CHECK_QUESTION") or "¿Cuál es la diferencia principal entre una lista enlazada y un array?"
    rationale = _extract_field(raw_answer, "RATIONALE") or "Comparar ambas estructuras ayuda a reducir la carga cognitiva en un nivel inicial."

    mode = mode.lower().strip()
    focus = focus.lower().strip()

    if mode not in VALID_MODES:
        mode = "compare_contrast"

    if focus not in VALID_FOCUS:
        focus = "definition"

    plan = {
        "mode": mode,
        "focus": focus,
        "check_question": check_question,
        "rationale": rationale,
    }

    print("[agent2][parsed_plan]", plan)

    return plan
