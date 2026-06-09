import os
import uuid
import fitz
from qdrant_client.models import PointStruct
from app.core.embeddings import embed_texts, get_embedding_model, get_device
from app.core.text_utils import clean_text, chunk_text
from app.core.retriever import ensure_collection, upsert_points

RAW_DIR = "/home/albertomatilla/rag_agentico_edu/data/raw"
BATCH_SIZE = 16
UPSERT_EVERY = 128

def extract_pages_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append((i + 1, text))
    return pages

def flush_points(buffer, total_inserted):
    if buffer:
        upsert_points(buffer)
        total_inserted += len(buffer)
        print(f"[qdrant] Insertados {len(buffer)} chunks en este lote. Total acumulado: {total_inserted}")
        buffer.clear()
    return total_inserted

def main():
    print("[ingest] Inicio de ingestión")
    model = get_embedding_model()
    device = get_device()
    vector_size = model.get_sentence_embedding_dimension()

    print(f"[ingest] Device embeddings: {device}")
    print(f"[ingest] Dimensión de embedding: {vector_size}")

    ensure_collection(vector_size)

    files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")])
    print(f"[ingest] PDFs detectados: {len(files)}")

    total_inserted = 0
    point_buffer = []

    for file_idx, filename in enumerate(files, start=1):
        pdf_path = os.path.join(RAW_DIR, filename)
        print(f"\n[ingest] ({file_idx}/{len(files)}) Procesando: {filename}")

        try:
            pages = extract_pages_from_pdf(pdf_path)
            print(f"[ingest] Páginas con texto: {len(pages)}")
        except Exception as e:
            print(f"[error] No se pudo leer {filename}: {e}")
            continue

        book_chunks = 0

        for page_num, page_text in pages:
            chunks = chunk_text(page_text, chunk_size=1200, overlap=200)
            if not chunks:
                continue

            try:
                vectors = embed_texts(chunks, batch_size=BATCH_SIZE)
            except Exception as e:
                print(f"[error] Embeddings fallaron en {filename}, página {page_num}: {e}")
                continue

            for chunk, vector in zip(chunks, vectors):
                chunk_id = str(uuid.uuid4())
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_file": filename,
                        "page": page_num,
                        "chunk_id": chunk_id,
                    }
                )
                point_buffer.append(point)
                book_chunks += 1

                if len(point_buffer) >= UPSERT_EVERY:
                    total_inserted = flush_points(point_buffer, total_inserted)

            if page_num % 20 == 0:
                print(f"[ingest] {filename}: página {page_num} procesada, chunks del libro hasta ahora: {book_chunks}")

        total_inserted = flush_points(point_buffer, total_inserted)
        print(f"[ingest] Libro completado: {filename}. Chunks indexados del libro: {book_chunks}")

    total_inserted = flush_points(point_buffer, total_inserted)
    print(f"\n[ingest] Finalizado. Total de chunks indexados: {total_inserted}")

if __name__ == "__main__":
    main()
