import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=str(ENV_PATH))

@dataclass
class Settings:
    qdrant_mode: str = os.getenv("QDRANT_MODE", "local")
    qdrant_path: str = os.getenv("QDRANT_PATH", str(BASE_DIR / "data" / "index" / "qdrant_local"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "books_cs_v1")

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

    qwen_model: str = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8010"))

settings = Settings()
