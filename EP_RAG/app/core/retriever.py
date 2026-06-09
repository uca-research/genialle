from qdrant_client.models import Distance, VectorParams
from app.core.qdrant_factory import get_qdrant_client
from app.core.config import settings
from app.core.embeddings import embed_query

def ensure_collection(vector_size: int):
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if settings.qdrant_collection not in names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def upsert_points(points):
    client = get_qdrant_client()
    client.upsert(collection_name=settings.qdrant_collection, points=points)

def search(question: str, top_k: int = 5):
    client = get_qdrant_client()
    query_vector = embed_query(question)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k
        )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append({
            "text": payload.get("text", ""),
            "score": float(hit.score),
            "metadata": {
                "source_file": payload.get("source_file", "desconocido"),
                "page": payload.get("page", -1),
                "chunk_id": payload.get("chunk_id", "desconocido")
            }
        })
    return results
