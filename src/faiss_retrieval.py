import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS = "processed_data/chunks_500_standard.json"
EMBEDDINGS = "processed_data/embeddings_500_standard.npy"
OUTPUT = "processed_data/faiss_retrieval_results.json"

QUESTIONS = [
    "What are the common symptoms and signs of cancer?",
    "What are the side effects of chemotherapy?",
    "How is stomach cancer diagnosed?",
    "What treatments are available for stomach cancer?",
    "What is radiation therapy?",
    "What are the side effects of radiation therapy?",
    "What is melanoma?",
    "How is melanoma diagnosed?",
    "What is osteosarcoma?",
    "How is osteosarcoma treated?",
    "What are the risk factors for cervical cancer?",
    "What is the role of biopsy in cancer diagnosis?",
    "What is cancer staging?",
    "What is chemotherapy?",
    "What are supportive care measures for cancer patients?"
]

def main():
    with open(CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    results = {}

    for question in QUESTIONS:
        q = model.encode([question], convert_to_numpy=True).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)

        start = time.perf_counter()
        scores, indices = index.search(q, 3)
        latency = (time.perf_counter() - start) * 1000

        results[question] = {
            "latency_ms": latency,
            "results": []
        }

        for rank, (score, i) in enumerate(zip(scores[0], indices[0]), 1):
            results[question]["results"].append({
                "rank": rank,
                "chunk_id": chunks[i]["chunk_id"],
                "document_title": chunks[i]["document_title"],
                "page": chunks[i]["page_start"],
                "score": float(score),
                "text": chunks[i]["text"]
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("FAISS RETRIEVAL COMPLETE")
    print("Chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()