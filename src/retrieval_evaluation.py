import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    100: "processed_data/chunks_100.json",
    200: "processed_data/oncology_chunks.json",
    300: "processed_data/chunks_300.json",
    500: "processed_data/chunks_500.json"
}

EMBEDDINGS = {
    100: "processed_data/embeddings_100.npy",
    200: "processed_data/embeddings.npy",
    300: "processed_data/embeddings_300.npy",
    500: "processed_data/embeddings_500.npy"
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 80)
print("ONCOLOGY RETRIEVAL EVALUATION")
print("=" * 80)

print("\nLoading model...")
model = SentenceTransformer(MODEL_NAME)

print("Model:", MODEL_NAME)
print("Number of questions:", len(QUESTIONS))
print("Top-k:", TOP_K)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_embedding, document_embeddings):

    dot_products = np.dot(
        document_embeddings,
        query_embedding
    )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    scores = dot_products / (
        document_norms * query_norm
    )

    return scores


# ============================================================
# LOAD ALL DATASETS
# ============================================================

datasets = {}
embeddings = {}

for chunk_size in DATASETS:

    print("\nLoading", chunk_size, "word dataset...")

    with open(
        DATASETS[chunk_size],
        "r",
        encoding="utf-8"
    ) as f:

        datasets[chunk_size] = json.load(f)

    embeddings[chunk_size] = np.load(
        EMBEDDINGS[chunk_size]
    )

    print(
        "Chunks:",
        len(datasets[chunk_size])
    )

    print(
        "Embedding shape:",
        embeddings[chunk_size].shape
    )


# ============================================================
# GENERATE QUERY EMBEDDINGS
# ============================================================

print("\nGenerating query embeddings...")

query_embeddings = model.encode(
    QUESTIONS,
    convert_to_numpy=True,
    normalize_embeddings=False
)

print(
    "Query embedding shape:",
    query_embeddings.shape
)


# ============================================================
# RETRIEVAL
# ============================================================

results = {}

for question_index, question in enumerate(QUESTIONS):

    print("\n" + "=" * 80)
    print("QUESTION", question_index + 1)
    print(question)
    print("=" * 80)

    query_embedding = query_embeddings[question_index]

    results[question] = {}

    for chunk_size in DATASETS:

        scores = cosine_similarity(
            query_embedding,
            embeddings[chunk_size]
        )

        top_indices = np.argsort(scores)[::-1][:TOP_K]

        retrieved = []

        for rank, index in enumerate(top_indices):

            chunk = datasets[chunk_size][index]

            result = {
                "rank": rank + 1,
                "index": int(index),
                "score": float(scores[index]),
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document_title"],
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "text": chunk["text"]
            }

            retrieved.append(result)

        results[question][chunk_size] = retrieved

        print("\n---", chunk_size, "WORD CHUNKS ---")

        for result in retrieved:

            print(
                "\nRank:",
                result["rank"]
            )

            print(
                "Score:",
                round(result["score"], 4)
            )

            print(
                "Document:",
                result["document"]
            )

            print(
                "Page:",
                result["page_start"],
                "-",
                result["page_end"]
            )

            print(
                "Chunk ID:",
                result["chunk_id"]
            )

            print(
                "Text:",
                result["text"][:300],
                "..."
            )


# ============================================================
# SAVE RAW RETRIEVAL RESULTS
# ============================================================

output_file = "processed_data/retrieval_results.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION COMPLETE")
print("=" * 80)

print(
    "\nRaw retrieval results saved to:"
)

print(output_file)

print(
    """
IMPORTANT:
Precision@3 will be calculated after relevance
labels are assigned to the retrieved chunks.
"""
)