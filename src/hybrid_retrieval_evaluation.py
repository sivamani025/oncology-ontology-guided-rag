import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "processed_data"

CHUNK_FILE = os.path.join(
    DATA_DIR,
    "chunks_500_standard.json"
)

EMBEDDING_FILE = os.path.join(
    DATA_DIR,
    "embeddings_500_standard.npy"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "hybrid_retrieval_results.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3

# Semantic weight
ALPHA = 0.7


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
# LOAD DATA
# ============================================================

def load_chunks():

    if not os.path.exists(CHUNK_FILE):
        raise FileNotFoundError(
            f"Chunk file not found:\n{CHUNK_FILE}"
        )

    with open(
        CHUNK_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    return chunks


def load_embeddings():

    if not os.path.exists(EMBEDDING_FILE):
        raise FileNotFoundError(
            f"Embedding file not found:\n{EMBEDDING_FILE}"
        )

    embeddings = np.load(
        EMBEDDING_FILE
    )

    return embeddings


# ============================================================
# NORMALIZE SCORES
# ============================================================

def min_max_normalize(scores):

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# SEMANTIC COSINE SIMILARITY
# ============================================================

def semantic_similarity(
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

    denominator = (
        document_norms * query_norm
    )

    denominator = np.where(
        denominator == 0,
        1e-10,
        denominator
    )

    return dot_products / denominator


# ============================================================
# TF-IDF LEXICAL SIMILARITY
# ============================================================

def lexical_similarity(
    vectorizer,
    document_matrix,
    query
):

    query_vector = vectorizer.transform(
        [query]
    )

    scores = (
        document_matrix
        @ query_vector.T
    )

    return np.asarray(
        scores.toarray()
    ).flatten()


# ============================================================
# GET TOP-K
# ============================================================

def get_top_k(scores):

    indices = np.argsort(
        scores
    )[::-1][:TOP_K]

    return indices


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HYBRID LEXICAL + SEMANTIC RETRIEVAL")
    print("=" * 80)

    print(
        f"\nChunk configuration : 500 words"
    )

    print(
        f"Overlap             : 100 words"
    )

    print(
        f"Top-K               : {TOP_K}"
    )

    print(
        f"Semantic weight     : {ALPHA}"
    )

    print(
        f"Lexical weight      : {1 - ALPHA}"
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"\nNumber of chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Load semantic embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    if embeddings.shape != (
        len(chunks),
        384
    ):

        raise ValueError(
            f"Embedding shape mismatch: "
            f"{embeddings.shape}"
        )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Load MiniLM
    # --------------------------------------------------------

    print(
        "\nLoading MiniLM model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Prepare documents
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Build TF-IDF index
    # --------------------------------------------------------

    print(
        "\nBuilding TF-IDF index..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    document_matrix = vectorizer.fit_transform(
        texts
    )

    print(
        f"TF-IDF matrix shape: "
        f"{document_matrix.shape}"
    )

    # --------------------------------------------------------
    # Generate query embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating query embeddings..."
    )

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
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process every question
    # --------------------------------------------------------

    for question_number, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Question {question_number}: "
            f"{question}"
        )

        query_embedding = query_embeddings[
            question_number - 1
        ]

        # ----------------------------------------------------
        # Semantic scores
        # ----------------------------------------------------

        semantic_scores = semantic_similarity(
            query_embedding,
            embeddings
        )

        # ----------------------------------------------------
        # Lexical scores
        # ----------------------------------------------------

        lexical_scores = lexical_similarity(
            vectorizer,
            document_matrix,
            question
        )

        # ----------------------------------------------------
        # Normalize scores
        # ----------------------------------------------------

        semantic_normalized = min_max_normalize(
            semantic_scores
        )

        lexical_normalized = min_max_normalize(
            lexical_scores
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_scores = (
            ALPHA * semantic_normalized
            +
            (1 - ALPHA) * lexical_normalized
        )

        # ----------------------------------------------------
        # Get top-k
        # ----------------------------------------------------

        semantic_top = get_top_k(
            semantic_normalized
        )

        lexical_top = get_top_k(
            lexical_normalized
        )

        hybrid_top = get_top_k(
            hybrid_scores
        )

        # ----------------------------------------------------
        # Build retrieval result
        # ----------------------------------------------------

        question_result = {
            "question_number":
                question_number,

            "question":
                question,

            "semantic_only": [],

            "lexical_only": [],

            "hybrid": []
        }

        # ----------------------------------------------------
        # Semantic results
        # ----------------------------------------------------

        for rank, index in enumerate(
            semantic_top,
            start=1
        ):

            chunk = chunks[index]

            question_result[
                "semantic_only"
            ].append({

                "rank":
                    rank,

                "embedding_index":
                    int(index),

                "chunk_id":
                    chunk["chunk_id"],

                "document_title":
                    chunk["document_title"],

                "page_start":
                    chunk["page_start"],

                "similarity_score":
                    float(
                        semantic_scores[index]
                    ),

                "normalized_score":
                    float(
                        semantic_normalized[index]
                    ),

                "text":
                    chunk["text"]
            })

        # ----------------------------------------------------
        # Lexical results
        # ----------------------------------------------------

        for rank, index in enumerate(
            lexical_top,
            start=1
        ):

            chunk = chunks[index]

            question_result[
                "lexical_only"
            ].append({

                "rank":
                    rank,

                "embedding_index":
                    int(index),

                "chunk_id":
                    chunk["chunk_id"],

                "document_title":
                    chunk["document_title"],

                "page_start":
                    chunk["page_start"],

                "lexical_score":
                    float(
                        lexical_scores[index]
                    ),

                "normalized_score":
                    float(
                        lexical_normalized[index]
                    ),

                "text":
                    chunk["text"]
            })

        # ----------------------------------------------------
        # Hybrid results
        # ----------------------------------------------------

        for rank, index in enumerate(
            hybrid_top,
            start=1
        ):

            chunk = chunks[index]

            question_result[
                "hybrid"
            ].append({

                "rank":
                    rank,

                "embedding_index":
                    int(index),

                "chunk_id":
                    chunk["chunk_id"],

                "document_title":
                    chunk["document_title"],

                "page_start":
                    chunk["page_start"],

                "semantic_score":
                    float(
                        semantic_scores[index]
                    ),

                "lexical_score":
                    float(
                        lexical_scores[index]
                    ),

                "hybrid_score":
                    float(
                        hybrid_scores[index]
                    ),

                "text":
                    chunk["text"]
            })

        results.append(
            question_result
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "configuration": {

            "chunk_size_words":
                500,

            "overlap_words":
                100,

            "top_k":
                TOP_K,

            "semantic_weight":
                ALPHA,

            "lexical_weight":
                1 - ALPHA,

            "number_of_questions":
                len(QUESTIONS),

            "number_of_chunks":
                len(chunks)
        },

        "questions":
            results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("HYBRID RETRIEVAL COMPLETE")
    print("=" * 80)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nMethods evaluated:")

    print(
        "1. Semantic-only retrieval"
    )

    print(
        "2. TF-IDF lexical retrieval"
    )

    print(
        "3. Hybrid retrieval"
    )

    print("\nNext step:")
    print(
        "Assign relevance labels and calculate "
        "Precision@3 for all three methods."
    )


if __name__ == "__main__":
    main()