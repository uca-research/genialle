import re

def _clean_source_name(source: str) -> str:
    source = re.sub(r",\s*pág\.\s*\d+\s*$", "", source, flags=re.IGNORECASE)
    return source.strip()

def _src(ev: dict) -> str:
    source = _clean_source_name(ev.get("source", "Fuente desconocida"))
    page = ev.get("page", "?")
    return f"{source}, pág. {page}"

def _top_sources(agent1_payload: dict):
    ev = agent1_payload.get("retrieved_evidence", [])
    if not ev:
        return ("Fuente desconocida", "Fuente desconocida", "Fuente desconocida")

    s1 = _src(ev[0]) if len(ev) > 0 else "Fuente desconocida"
    s2 = _src(ev[1]) if len(ev) > 1 else s1
    s3 = _src(ev[2]) if len(ev) > 2 else s2
    return s1, s2, s3

def _build_steps(mode: str, s1: str, s2: str, s3: str):
    if mode == "worked_example":
        return [
            f"Una lista enlazada es una secuencia de nodos. Cada nodo guarda un valor y una referencia al siguiente nodo. Por eso los elementos no necesitan estar colocados de forma contigua en memoria ({s3}).",
            f"Un array, en cambio, guarda sus elementos en memoria contigua. Eso hace muy rápido el acceso directo por posición, pero puede exigir reservar un bloque continuo suficientemente grande ({s1}).",
            f"Ejemplo mental: si quieres leer rápidamente el elemento de una posición concreta, el array suele ser más cómodo; si ya estás situado en un nodo concreto y quieres insertar o borrar cerca de él, la lista enlazada puede ser ventajosa ({s3}).",
        ]

    if mode == "misconception_first":
        return [
            f"Un error frecuente es pensar que una lista enlazada es simplemente un array más flexible. No es así: su estructura interna es distinta, porque está formada por nodos enlazados entre sí ({s3}).",
            f"En un array, los elementos están en memoria contigua y eso favorece el acceso rápido por índice ({s1}). En una lista enlazada, para llegar a un elemento concreto normalmente hay que recorrer la secuencia nodo a nodo ({s3}).",
            f"La diferencia práctica es esta: el array suele ser mejor para acceso directo rápido; la lista enlazada puede ser útil cuando interesa modificar la secuencia mediante enlaces, especialmente cerca de un nodo ya localizado ({s2}; {s3}).",
        ]

    return [
        f"Una lista enlazada es una estructura formada por nodos; cada nodo guarda un dato y una referencia al siguiente. Por eso la secuencia se recorre enlazando nodos unos con otros ({s3}).",
        f"Un array guarda los elementos en memoria contigua. Esa disposición permite acceder muy rápido a una posición concreta, porque su localización depende del índice ({s1}).",
        f"La diferencia principal es el compromiso entre acceso y flexibilidad estructural: el array suele favorecer el acceso directo, mientras que la lista enlazada permite reorganizar enlaces entre nodos sin depender de memoria contigua ({s1}; {s3}).",
    ]

def _build_example(s1: str, s3: str):
    return (
        "Imagina que quieres almacenar los valores 10, 20 y 30.\n"
        "- En un array, podrías pensar en tres huecos contiguos en memoria: [10, 20, 30].\n"
        "- En una lista enlazada, tendrías tres nodos conectados: 10 -> 20 -> 30.\n"
        "Si quisieras leer directamente el segundo elemento, el array lo facilita mucho. "
        "Si quisieras insertar un nuevo nodo entre otros dos nodos ya localizados, la lista enlazada puede resultar más natural desde el punto de vista estructural "
        f"({s1}; {s3})."
    )

def _idea_clave():
    return (
        "La idea clave no es que una estructura sea mejor en general, sino que array y lista enlazada resuelven necesidades distintas."
    )

def _error_tipico():
    return (
        "Pensar que la lista enlazada siempre sustituye al array o que siempre ahorra memoria. La elección depende del tipo de acceso y modificación que necesites."
    )

def render_pedagogical_answer(agent1_payload: dict, agent2_plan: dict) -> str:
    s1, s2, s3 = _top_sources(agent1_payload)

    mode = agent2_plan.get("mode", "compare_contrast")
    check_question = agent2_plan.get(
        "check_question",
        "¿Qué ventaja principal tiene un array frente a una lista enlazada?"
    )
    rationale = agent2_plan.get(
        "rationale",
        "Comparar ambas estructuras ayuda a organizar la información sin sobrecargar al estudiante."
    )

    steps = _build_steps(mode, s1, s2, s3)
    example = _build_example(s1, s3)
    idea = _idea_clave()
    error = _error_tipico()

    return (
        "respuesta_para_estudiante:\n"
        f"1. {steps[0]}\n"
        f"2. {steps[1]}\n"
        f"3. {steps[2]}\n\n"
        "ejemplo_trabajado:\n"
        f"{example}\n\n"
        "idea_clave:\n"
        f"{idea}\n\n"
        "error_tipico:\n"
        f"{error}\n\n"
        "pregunta_de_comprobacion:\n"
        f"{check_question}\n\n"
        "notas_didacticas:\n"
        f"- modo_pedagogico: {mode}\n"
        f"- razon_principal: {rationale}"
    )
