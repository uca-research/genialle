import torch
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None
_device = None

def get_embedding_model():
    global _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embeddings] Cargando modelo {settings.embed_model} en device={_device}")
        _model = SentenceTransformer(settings.embed_model, device=_device)
    return _model

def get_device():
    global _device
    if _device is None:
        get_embedding_model()
    return _device

def embed_texts(texts, batch_size=16):
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

def embed_query(text):
    model = get_embedding_model()
    return model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()[0]
