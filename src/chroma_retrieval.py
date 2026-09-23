import json
import time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS = "processed_data/chunks_500_standard.json"
EMBEDDINGS = "processed_data/embeddings_500_standard.npy"
OUTPUT = "processed_data/chroma_retrieval_results.json"

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
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="processed_data/chroma_db")

    try:
        client.delete_collection("oncology")
    except:
        pass

    collection = client.create_collection(
        name="oncology",
        configuration={"hnsw": {"space": "cosine"}}
    )

    collection.add(
        ids=[x["chunk_id"] for x in chunks],
        embeddings=embeddings.tolist(),
        documents=[x["text"] for x in chunks],
        metadatas=[
            {
                "document": str(x["document_title"]),
                "page": str(x["page_start"])
            }
            for x in chunks
        ]
    )

    results = {}

    for question in QUESTIONS:
        query = model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")[0].tolist()

        start = time.perf_counter()

        r = collection.query(
            query_embeddings=[query],
            n_results=3
        )

        latency = (time.perf_counter() - start) * 1000

        results[question] = {
            "latency_ms": latency,
            "results": []
        }

        for rank in range(3):
            results[question]["results"].append({
                "rank": rank + 1,
                "chunk_id": r["ids"][0][rank],
                "document_title": r["metadatas"][0][rank]["document"],
                "page": r["metadatas"][0][rank]["page"],
                "score": r["distances"][0][rank],
                "text": r["documents"][0][rank]
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("CHROMADB RETRIEVAL COMPLETE")
    print("Chunks:", len(chunks))
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()