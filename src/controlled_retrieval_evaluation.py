import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = "processed_data"

CHUNK_SIZES = [100, 200, 300, 500]

TOP_K = 3

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "controlled_retrieval_results.json"
)


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
# LOAD CHUNKS
# ============================================================

def load_chunks(chunk_size):

    file_path = os.path.join(
        DATA_DIR,
        f"chunks_{chunk_size}_standard.json"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Chunk file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    return chunks


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(chunk_size):

    file_path = os.path.join(
        DATA_DIR,
        f"embeddings_{chunk_size}_standard.npy"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Embedding file not found: {file_path}"
        )

    embeddings = np.load(file_path)

    return embeddings


# ============================================================
# MANUAL COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_embedding,
    document_embeddings
):

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

    # Prevent division by zero
    denominator = (
        document_norms * query_norm
    )

    denominator = np.where(
        denominator == 0,
        1e-10,
        denominator
    )

    scores = dot_products / denominator

    return scores


# ============================================================
# RETRIEVE TOP-K
# ============================================================

def retrieve_top_k(
    query_embedding,
    embeddings,
    top_k
):

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        results.append({
            "rank": rank,
            "embedding_index": int(index),
            "similarity_score": float(
                scores[index]
            )
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONTROLLED RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"\nModel      : {MODEL_NAME}")
    print(f"Questions  : {len(QUESTIONS)}")
    print(f"Top-K      : {TOP_K}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Generate query embeddings
    # --------------------------------------------------------

    print("\nGenerating query embeddings...")

    query_embeddings = model.encode(
        QUESTIONS,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype(np.float32)

    print(
        f"Query embedding shape: "
        f"{query_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Store all results
    # --------------------------------------------------------

    all_results = {}

    # --------------------------------------------------------
    # Evaluate each chunk size
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        print("\n" + "=" * 80)
        print(
            f"EVALUATING {chunk_size}-WORD CHUNKS"
        )
        print("=" * 80)

        chunks = load_chunks(
            chunk_size
        )

        embeddings = load_embeddings(
            chunk_size
        )

        # ----------------------------------------------------
        # Validate embedding matrix
        # ----------------------------------------------------

        if embeddings.shape != (
            len(chunks),
            384
        ):

            raise ValueError(
                f"Embedding shape mismatch for "
                f"{chunk_size}-word chunks.\n"
                f"Chunks: {len(chunks)}\n"
                f"Embeddings: {embeddings.shape}"
            )

        print(
            f"Chunks     : {len(chunks)}"
        )

        print(
            f"Embeddings : {embeddings.shape}"
        )

        question_results = []

        # ----------------------------------------------------
        # Process questions
        # ----------------------------------------------------

        for question_number, question in enumerate(
            QUESTIONS,
            start=1
        ):

            query_embedding = (
                query_embeddings[
                    question_number - 1
                ]
            )

            retrieved = retrieve_top_k(
                query_embedding,
                embeddings,
                TOP_K
            )

            # Attach chunk information
            detailed_results = []

            for result in retrieved:

                index = result[
                    "embedding_index"
                ]

                chunk = chunks[index]

                detailed_results.append({
                    "rank": result["rank"],
                    "embedding_index": index,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document_title": chunk[
                        "document_title"
                    ],
                    "source_file": chunk[
                        "source_file"
                    ],
                    "page_start": chunk[
                        "page_start"
                    ],
                    "page_end": chunk[
                        "page_end"
                    ],
                    "similarity_score": result[
                        "similarity_score"
                    ],
                    "text": chunk["text"]
                })

            question_results.append({
                "question_number": question_number,
                "question": question,
                "top_k_results": detailed_results
            })

        all_results[str(chunk_size)] = {
            "chunk_size_words": chunk_size,
            "number_of_chunks": len(chunks),
            "top_k": TOP_K,
            "questions": question_results
        }

        print(
            f"Completed {len(QUESTIONS)} questions."
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONTROLLED RETRIEVAL EVALUATION COMPLETE")
    print("=" * 80)

    print(
        "\nRaw retrieval results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nConfigurations evaluated:")

    for chunk_size in CHUNK_SIZES:

        print(
            f"  {chunk_size} words → "
            f"{len(all_results[str(chunk_size)]['questions'])} questions"
        )

    print("\nImportant:")
    print(
        "Precision@3 requires manual relevance labels "
        "for the retrieved top-3 chunks."
    )

    print(
        "\nNext step:"
    )

    print(
        "Inspect the retrieved chunks and assign "
        "relevance labels (1 = relevant, 0 = not relevant)."
    )


if __name__ == "__main__":
    main()