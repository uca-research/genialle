import json
from app.core.local_llm_runtime import generate_qwen_chat

def build_agent1_prompt(question: str, learner_level: str, retrieved_chunks: list) -> str:
    evidence_text = "\n\n".join([
        f"[Fuente: {c['metadata']['source_file']}, pág. {c['metadata']['page']}, score={c['score']:.4f}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    return f"""
Actúa como curador epistémico en un sistema RAG educativo de informática.

Tu tarea:
1. Analizar la pregunta del estudiante.
2. Revisar la evidencia recuperada.
3. Extraer solo la evidencia relevante.
4. Detectar prerrequisitos y huecos de comprensión.
5. Devolver EXCLUSIVAMENTE un JSON válido.

Pregunta:
{question}

Nivel del estudiante:
{learner_level}

Evidencia:
{evidence_text}

Devuelve SOLO este esquema JSON:
{{
  "user_query": "...",
  "learner_level_estimate": "...",
  "prerequisites": ["..."],
  "knowledge_gaps": ["..."],
  "pedagogical_constraints": {{
    "max_new_concepts": 3,
    "needs_worked_example": true,
    "needs_comparison_table": false
  }},
  "retrieved_evidence": [
    {{
      "source": "...",
      "page": 0,
      "concept": "...",
      "snippet": "...",
      "relevance": 0.0
    }}
  ]
}}
""".strip()

def run_agent1(question: str, learner_level: str, retrieved_chunks: list) -> dict:
    prompt = build_agent1_prompt(question, learner_level, retrieved_chunks)

    content = generate_qwen_chat(
        messages=[
            {"role": "system", "content": "Eres un curador epistémico riguroso. Devuelves únicamente JSON válido."},
            {"role": "user", "content": prompt}
        ],
        max_new_tokens=900,
        temperature=0.2
    )

    cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "user_query": question,
            "learner_level_estimate": learner_level,
            "prerequisites": [],
            "knowledge_gaps": ["Fallo de parseo JSON en agente 1"],
            "pedagogical_constraints": {
                "max_new_concepts": 3,
                "needs_worked_example": True,
                "needs_comparison_table": False
            },
            "retrieved_evidence": []
        }
