import json
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = os.path.join(
    "processed_data",
    "hybrid_retrieval_results.json"
)

OUTPUT_FILE = os.path.join(
    "processed_data",
    "hybrid_precision_results.json"
)

TOP_K = 3


# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_results(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# DISPLAY RETRIEVED CHUNK
# =============================================================================

def display_result(rank, result):
    print(f"\nRank {rank}")
    print("-" * 80)

    print("Chunk ID       :", result.get("chunk_id", "N/A"))
    print("Document       :", result.get("document_title", "N/A"))
    print("Page           :", result.get("page_start", "N/A"))

    if "similarity_score" in result:
        print(
            "Semantic score :",
            round(result["similarity_score"], 4)
        )

    if "lexical_score" in result:
        print(
            "Lexical score  :",
            round(result["lexical_score"], 4)
        )

    if "hybrid_score" in result:
        print(
            "Hybrid score   :",
            round(result["hybrid_score"], 4)
        )

    print("\nText:")
    print(result.get("text", ""))


# =============================================================================
# GET RELEVANCE LABEL
# =============================================================================

def get_label():

    while True:

        value = input(
            "\nIs this chunk relevant to the question? "
            "[1 = Relevant, 0 = Not relevant]: "
        ).strip()

        if value == "1":
            return 1

        if value == "0":
            return 0

        print("Please enter only 1 or 0.")


# =============================================================================
# LABEL ONE METHOD
# =============================================================================

def label_method(question, method_name, results):

    print("\n")
    print("=" * 100)
    print(f"METHOD: {method_name}")
    print("=" * 100)

    print("\nQuestion:")
    print(question)

    labels = []

    for rank, result in enumerate(results[:TOP_K], start=1):

        display_result(rank, result)

        label = get_label()

        labels.append({
            "rank": rank,
            "chunk_id": result.get("chunk_id"),
            "relevance": label
        })

    relevant_count = sum(
        item["relevance"] for item in labels
    )

    precision = relevant_count / TOP_K

    print("\n" + "-" * 80)
    print(f"{method_name} relevant chunks : {relevant_count}/{TOP_K}")
    print(f"{method_name} Precision@3     : {precision:.4f}")
    print("-" * 80)

    return labels, precision


# =============================================================================
# CALCULATE AVERAGE
# =============================================================================

def calculate_average(values):

    if len(values) == 0:
        return 0.0

    return sum(values) / len(values)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("HYBRID RETRIEVAL PRECISION@3 EVALUATION")
    print("=" * 100)

    print("\nInput file:")
    print(INPUT_FILE)

    if not os.path.exists(INPUT_FILE):

        print("\nERROR:")
        print("Input file was not found.")
        print(
            "Make sure hybrid_retrieval_evaluation.py "
            "has been executed first."
        )
        return

    # -------------------------------------------------------------------------
    # Load retrieval results
    # -------------------------------------------------------------------------

    data = load_results(INPUT_FILE)

    print("\nResults loaded successfully.")

    print("Number of questions:", len(data))
    print("Top-K:", TOP_K)

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    all_labels = {}

    semantic_precisions = []
    lexical_precisions = []
    hybrid_precisions = []

    # -------------------------------------------------------------------------
    # Process each question
    # -------------------------------------------------------------------------

    for question_number, question_data in enumerate(data.items(), start=1):

        question = question_data[0]
        results = question_data[1]

        print("\n\n")
        print("#" * 100)
        print(
            f"QUESTION {question_number} / {len(data)}"
        )
        print("#" * 100)

        print("\nQuestion:")
        print(question)

        # =====================================================================
        # IMPORTANT:
        # The JSON may use different names depending on the retrieval script.
        # We try the expected names first.
        # =====================================================================

        semantic_results = (
            results.get("semantic", [])
        )

        lexical_results = (
            results.get("lexical", [])
        )

        hybrid_results = (
            results.get("hybrid", [])
        )

        # ---------------------------------------------------------------------
        # If the JSON uses alternative names, check those too.
        # ---------------------------------------------------------------------

        if not semantic_results:
            semantic_results = results.get(
                "semantic_only",
                results.get("Semantic-only", [])
            )

        if not lexical_results:
            lexical_results = results.get(
                "lexical_only",
                results.get("tfidf", [])
            )

        if not hybrid_results:
            hybrid_results = results.get(
                "hybrid_retrieval",
                results.get("Hybrid", [])
            )

        # ---------------------------------------------------------------------
        # Label Semantic
        # ---------------------------------------------------------------------

        semantic_labels, semantic_precision = label_method(
            question,
            "Semantic-only",
            semantic_results
        )

        semantic_precisions.append(
            semantic_precision
        )

        # ---------------------------------------------------------------------
        # Label Lexical
        # ---------------------------------------------------------------------

        lexical_labels, lexical_precision = label_method(
            question,
            "TF-IDF / Lexical-only",
            lexical_results
        )

        lexical_precisions.append(
            lexical_precision
        )

        # ---------------------------------------------------------------------
        # Label Hybrid
        # ---------------------------------------------------------------------

        hybrid_labels, hybrid_precision = label_method(
            question,
            "Hybrid",
            hybrid_results
        )

        hybrid_precisions.append(
            hybrid_precision
        )

        # ---------------------------------------------------------------------
        # Store question results
        # ---------------------------------------------------------------------

        all_labels[question] = {

            "semantic": {
                "labels": semantic_labels,
                "precision_at_3": semantic_precision
            },

            "lexical": {
                "labels": lexical_labels,
                "precision_at_3": lexical_precision
            },

            "hybrid": {
                "labels": hybrid_labels,
                "precision_at_3": hybrid_precision
            }
        }

    # =========================================================================
    # FINAL AVERAGES
    # =========================================================================

    average_semantic = calculate_average(
        semantic_precisions
    )

    average_lexical = calculate_average(
        lexical_precisions
    )

    average_hybrid = calculate_average(
        hybrid_precisions
    )

    # =========================================================================
    # PRINT FINAL RESULTS
    # =========================================================================

    print("\n\n")
    print("=" * 100)
    print("FINAL PRECISION@3 RESULTS")
    print("=" * 100)

    print(
        f"\nSemantic-only Precision@3 : "
        f"{average_semantic:.4f}"
    )

    print(
        f"TF-IDF Precision@3         : "
        f"{average_lexical:.4f}"
    )

    print(
        f"Hybrid Precision@3        : "
        f"{average_hybrid:.4f}"
    )

    # -------------------------------------------------------------------------
    # Improvement calculations
    # -------------------------------------------------------------------------

    hybrid_vs_semantic = (
        average_hybrid - average_semantic
    )

    hybrid_vs_lexical = (
        average_hybrid - average_lexical
    )

    print("\n")
    print("-" * 100)

    print(
        f"Hybrid - Semantic          : "
        f"{hybrid_vs_semantic:+.4f}"
    )

    print(
        f"Hybrid - TF-IDF            : "
        f"{hybrid_vs_lexical:+.4f}"
    )

    # -------------------------------------------------------------------------
    # Determine best method
    # -------------------------------------------------------------------------

    scores = {
        "Semantic-only": average_semantic,
        "TF-IDF": average_lexical,
        "Hybrid": average_hybrid
    }

    best_method = max(
        scores,
        key=scores.get
    )

    print("\nBest retrieval method:")
    print(
        f"{best_method} "
        f"(Precision@3 = {scores[best_method]:.4f})"
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    final_results = {

        "experiment": "Hybrid Lexical + Semantic Retrieval",

        "chunk_size_words": 500,

        "overlap_words": 100,

        "top_k": TOP_K,

        "semantic_weight": 0.7,

        "lexical_weight": 0.3,

        "number_of_questions": len(data),

        "precision_at_3": {

            "semantic_only": average_semantic,

            "tfidf_lexical_only": average_lexical,

            "hybrid": average_hybrid
        },

        "difference": {

            "hybrid_minus_semantic":
                hybrid_vs_semantic,

            "hybrid_minus_tfidf":
                hybrid_vs_lexical
        },

        "best_method": best_method,

        "per_question": all_labels
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nFinal summary:")
    print(
        f"Semantic-only : {average_semantic:.4f}"
    )
    print(
        f"TF-IDF        : {average_lexical:.4f}"
    )
    print(
        f"Hybrid        : {average_hybrid:.4f}"
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
    