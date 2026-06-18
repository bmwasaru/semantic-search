import numpy as np

from sentence_transformers import SentenceTransformer

from config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


def semantic_search(query, embeddings, chunks, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []

    for idx in top_indices:

        results.append({
            "page": chunks[idx]["page"],
            "score": float(scores[idx]),
            "text": chunks[idx]["text"]
        })

    return results