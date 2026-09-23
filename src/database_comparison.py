import json
import time
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS = "processed_data/chunks_500_standard.json"
EMBEDDINGS = "processed_data/embeddings_500_standard.npy"
OUTPUT = "processed_data/database_comparison_results.json"

TOP_K = 3

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


def cosine(q, x):
    return np.dot(x, q) / (
        np.linalg.norm(x, axis=1) *
        np.linalg.norm(q) + 1e-10
    )


def main():

    with open(CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(
        EMBEDDINGS
    ).astype("float32")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    queries = model.encode(
        QUESTIONS,
        convert_to_numpy=True
    ).astype("float32")

    # ---------------------------------------------------------
    # NumPy
    # ---------------------------------------------------------

    numpy_times = []
    numpy_results = {}

    for question, q in zip(QUESTIONS, queries):

        start = time.perf_counter()

        scores = cosine(q, embeddings)
        indices = np.argsort(scores)[::-1][:TOP_K]

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        numpy_times.append(elapsed)

        numpy_results[question] = {
            "latency_ms": elapsed,
            "chunks": [
                chunks[i]["chunk_id"]
                for i in indices
            ]
        }

    # ---------------------------------------------------------
    # FAISS
    # ---------------------------------------------------------

    faiss_embeddings = embeddings.copy()

    faiss.normalize_L2(
        faiss_embeddings
    )

    index = faiss.IndexFlatIP(
        faiss_embeddings.shape[1]
    )

    index.add(
        faiss_embeddings
    )

    faiss_times = []
    faiss_results = {}

    for question, q in zip(QUESTIONS, queries):

        q = q.reshape(1, -1).copy()

        faiss.normalize_L2(q)

        start = time.perf_counter()

        scores, indices = index.search(
            q,
            TOP_K
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        faiss_times.append(elapsed)

        faiss_results[question] = {
            "latency_ms": elapsed,
            "chunks": [
                chunks[i]["chunk_id"]
                for i in indices[0]
            ]
        }

    # ---------------------------------------------------------
    # ChromaDB
    # ---------------------------------------------------------

    client = chromadb.PersistentClient(
        path="processed_data/chroma_db"
    )

    try:
        client.delete_collection("benchmark")
    except:
        pass

    collection = client.create_collection(
        name="benchmark",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    collection.add(
        ids=[
            x["chunk_id"]
            for x in chunks
        ],
        embeddings=embeddings.tolist(),
        documents=[
            x["text"]
            for x in chunks
        ],
        metadatas=[
            {
                "document": str(x["document_title"]),
                "page": str(x["page_start"])
            }
            for x in chunks
        ]
    )

    chroma_times = []
    chroma_results = {}

    for question, q in zip(QUESTIONS, queries):

        start = time.perf_counter()

        result = collection.query(
            query_embeddings=[
                q.tolist()
            ],
            n_results=TOP_K
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        chroma_times.append(elapsed)

        chroma_results[question] = {
            "latency_ms": elapsed,
            "chunks": result["ids"][0]
        }

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "NumPy": {
            "average_latency_ms":
                float(np.mean(numpy_times)),
            "results": numpy_results
        },

        "FAISS": {
            "average_latency_ms":
                float(np.mean(faiss_times)),
            "results": faiss_results
        },

        "ChromaDB": {
            "average_latency_ms":
                float(np.mean(chroma_times)),
            "results": chroma_results
        }
    }

    output = {
        "configuration": {
            "documents": 11,
            "chunks": len(chunks),
            "chunk_size": 500,
            "overlap": 100,
            "embedding_dimension": 384,
            "queries": len(QUESTIONS),
            "top_k": TOP_K
        },
        "comparison": summary
    }

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATABASE RETRIEVAL BENCHMARK")
    print("=" * 70)

    print(
        f"\nChunks     : {len(chunks)}"
    )
    print(
        f"Queries    : {len(QUESTIONS)}"
    )
    print(
        f"Embedding  : 384"
    )
    print(
        f"Top-K      : {TOP_K}"
    )

    print("\n" + "-" * 70)
    print(
        f"{'Method':<15} {'Avg Latency (ms)':>20}"
    )
    print("-" * 70)

    print(
        f"{'NumPy':<15} "
        f"{np.mean(numpy_times):>20.4f}"
    )

    print(
        f"{'FAISS':<15} "
        f"{np.mean(faiss_times):>20.4f}"
    )

    print(
        f"{'ChromaDB':<15} "
        f"{np.mean(chroma_times):>20.4f}"
    )

    print("\nResults saved to:")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("DATABASE BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
