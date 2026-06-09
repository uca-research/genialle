import os
from qdrant_client import QdrantClient

def get_qdrant_client():
    mode = os.getenv("QDRANT_MODE", "local").lower()

    if mode == "local":
        path = os.getenv("QDRANT_PATH", "./data/index/qdrant_local")
        return QdrantClient(path=path)

    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)
    return QdrantClient(url=url, api_key=api_key)
