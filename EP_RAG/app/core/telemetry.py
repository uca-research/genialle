import json
import os
import time
import uuid
from datetime import datetime, timezone

import psutil
import torch

LOG_DIR = "/home/albertomatilla/rag_agentico_edu/logs"
LOG_FILE = os.path.join(LOG_DIR, "queries.jsonl")

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def new_request_id():
    return str(uuid.uuid4())

def process_memory_mb():
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024)
    return round(rss, 2)

def cuda_metrics():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_max_allocated_mb": 0.0,
            "cuda_max_reserved_mb": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    return {
        "cuda_available": True,
        "cuda_allocated_mb": round(allocated, 2),
        "cuda_reserved_mb": round(reserved, 2),
        "cuda_max_allocated_mb": round(max_allocated, 2),
        "cuda_max_reserved_mb": round(max_reserved, 2),
    }

def reset_cuda_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def timer_start():
    return time.perf_counter()

def timer_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)

def append_jsonl(record: dict):
    ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
