import os
import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    MODEL_NAME,
    PDF_PATH,
    EMBEDDINGS_PATH,
    CHUNKS_PATH
)

from services.pdf_loader import (
    extract_text_from_pdf,
    chunk_text
)


model = SentenceTransformer(MODEL_NAME)


def build_embeddings():
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(pages)
    
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))

    return embeddings, chunks


def load_embeddings():

    if (
        os.path.exists(EMBEDDINGS_PATH)
        and os.path.exists(CHUNKS_PATH)
    ):
        embeddings = np.load(EMBEDDINGS_PATH)

        chunks = np.load(
            CHUNKS_PATH,
            allow_pickle=True
        ).tolist()

        return embeddings, chunks

    return build_embeddings()