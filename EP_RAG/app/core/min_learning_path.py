import re

def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = _clean_text(item)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

def build_minimum_learning_path(agent1_payload: dict) -> dict:
    constraints = agent1_payload.get("pedagogical_constraints", {})
    max_new_concepts = constraints.get("max_new_concepts", 3)

    prerequisites = _dedupe_keep_order(agent1_payload.get("prerequisites", []))
    knowledge_gaps = _dedupe_keep_order(agent1_payload.get("knowledge_gaps", []))

    knowledge_gaps = [
        g for g in knowledge_gaps
        if "fallo de parseo" not in g.lower()
    ]

    ordered_items = (prerequisites + knowledge_gaps)[:max_new_concepts]

    steps = []
    for idx, concept in enumerate(ordered_items, start=1):
        step_type = "prerequisite" if concept in prerequisites else "knowledge_gap"
        steps.append({
            "step": idx,
            "type": step_type,
            "concept": concept,
        })

    next_best_step = steps[0]["concept"] if steps else None

    return {
        "user_query": agent1_payload.get("user_query"),
        "learner_level_estimate": agent1_payload.get("learner_level_estimate"),
        "max_new_concepts": max_new_concepts,
        "ordered_steps": steps,
        "next_best_step": next_best_step,
        "path_length": len(steps),
    }
