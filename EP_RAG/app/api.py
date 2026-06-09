from fastapi import FastAPI
from dotenv import load_dotenv

from app.core.schemas import AskRequest
from app.core.retriever import search
from app.agents.agent1 import run_agent1
from app.agents.agent2 import run_agent2
from app.core.local_llm_runtime import unload_qwen, unload_deepseek
from app.core.pedagogical_renderer import render_pedagogical_answer
from app.core.telemetry import (
    now_iso,
    new_request_id,
    process_memory_mb,
    cuda_metrics,
    reset_cuda_peak_memory,
    timer_start,
    timer_ms,
    append_jsonl,
)

load_dotenv(dotenv_path="/home/albertomatilla/rag_agentico_edu/.env")

app = FastAPI(title="RAG agéntico educativo")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rag_agentico_educativo"
    }

@app.post("/ask")
def ask(req: AskRequest):
    request_id = new_request_id()
    started_total = timer_start()

    reset_cuda_peak_memory()

    started_retrieval = timer_start()
    retrieved = search(req.question, top_k=req.top_k)
    retrieval_ms = timer_ms(started_retrieval)

    started_agent1 = timer_start()
    agent1_payload = run_agent1(req.question, req.learner_level, retrieved)
    agent1_ms = timer_ms(started_agent1)

    unload_qwen()

    started_agent2 = timer_start()
    agent2_plan = run_agent2(agent1_payload)
    agent2_ms = timer_ms(started_agent2)

    final_answer = render_pedagogical_answer(agent1_payload, agent2_plan)

    unload_deepseek()

    total_ms = timer_ms(started_total)

    mem = {
        "process_rss_mb": process_memory_mb(),
        **cuda_metrics()
    }

    record = {
        "timestamp_utc": now_iso(),
        "request_id": request_id,
        "question": req.question,
        "learner_level": req.learner_level,
        "top_k": req.top_k,
        "retrieved_count": len(retrieved),
        "agent1_payload": agent1_payload,
        "agent2_plan": agent2_plan,
        "latency_ms": {
            "retrieval": retrieval_ms,
            "agent1": agent1_ms,
            "agent2": agent2_ms,
            "total": total_ms
        },
        "memory": mem
    }

    append_jsonl(record)

    return {
        "request_id": request_id,
        "question": req.question,
        "retrieved_count": len(retrieved),
        "agent1_payload": agent1_payload,
        "agent2_plan": agent2_plan,
        "latency_ms": {
            "retrieval": retrieval_ms,
            "agent1": agent1_ms,
            "agent2": agent2_ms,
            "total": total_ms
        },
        "memory": mem,
        "answer": final_answer
    }
