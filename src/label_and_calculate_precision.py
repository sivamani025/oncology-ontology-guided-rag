import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = os.path.join(
    "processed_data",
    "controlled_retrieval_results.json"
)

OUTPUT_FILE = os.path.join(
    "processed_data",
    "controlled_precision_results.json"
)

CHUNK_SIZES = [100, 200, 300, 500]

TOP_K = 3


# ============================================================
# LOAD RETRIEVAL RESULTS
# ============================================================

def load_results():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# GET RELEVANCE LABEL
# ============================================================

def get_label():

    while True:

        label = input(
            "\nEnter relevance "
            "(1 = relevant, 0 = not relevant): "
        ).strip()

        if label in ["0", "1"]:
            return int(label)

        print(
            "Invalid input. Please enter only 0 or 1."
        )


# ============================================================
# CALCULATE PRECISION@3
# ============================================================

def calculate_precision(labels):

    relevant = sum(labels)

    return relevant / TOP_K


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RELEVANCE LABELING AND PRECISION@3")
    print("=" * 80)

    results = load_results()

    labeled_results = {}

    # --------------------------------------------------------
    # Process each chunk size
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        size_key = str(chunk_size)

        size_data = results[size_key]

        print("\n")
        print("=" * 80)
        print(
            f"{chunk_size}-WORD CHUNKS"
        )
        print("=" * 80)

        question_results = []

        # ----------------------------------------------------
        # Process each question
        # ----------------------------------------------------

        for question_data in size_data["questions"]:

            question_number = question_data[
                "question_number"
            ]

            question = question_data[
                "question"
            ]

            retrieved = question_data[
                "top_k_results"
            ]

            print("\n")
            print("-" * 80)
            print(
                f"QUESTION {question_number}"
            )
            print("-" * 80)

            print(
                f"\n{question}\n"
            )

            labels = []

            # ------------------------------------------------
            # Show top-3 results
            # ------------------------------------------------

            for result in retrieved:

                rank = result["rank"]

                print("\n" + "." * 70)

                print(
                    f"RANK {rank}"
                )

                print(
                    f"Similarity : "
                    f"{result['similarity_score']:.4f}"
                )

                print(
                    f"Document   : "
                    f"{result['document_title']}"
                )

                print(
                    f"Page       : "
                    f"{result['page_start']}"
                )

                print(
                    f"Chunk ID   : "
                    f"{result['chunk_id']}"
                )

                print("\nRetrieved text:")
                print(
                    result["text"]
                )

                # --------------------------------------------
                # Label result
                # --------------------------------------------

                label = get_label()

                labels.append(label)

            # ------------------------------------------------
            # Calculate Precision@3
            # ------------------------------------------------

            precision = calculate_precision(
                labels
            )

            print("\n")
            print(
                f"Labels       : {labels}"
            )

            print(
                f"Precision@3  : "
                f"{precision:.3f}"
            )

            # ------------------------------------------------
            # Store question result
            # ------------------------------------------------

            question_results.append({

                "question_number":
                    question_number,

                "question":
                    question,

                "labels":
                    labels,

                "relevant_count":
                    sum(labels),

                "precision_at_3":
                    precision,

                "retrieved_results":
                    retrieved
            })

        # ----------------------------------------------------
        # Average Precision@3
        # ----------------------------------------------------

        precisions = [
            q["precision_at_3"]
            for q in question_results
        ]

        average_precision = (
            sum(precisions)
            / len(precisions)
        )

        labeled_results[size_key] = {

            "chunk_size_words":
                chunk_size,

            "number_of_chunks":
                size_data["number_of_chunks"],

            "top_k":
                TOP_K,

            "questions":
                question_results,

            "average_precision_at_3":
                average_precision
        }

        # ----------------------------------------------------
        # Display size result
        # ----------------------------------------------------

        print("\n")
        print("=" * 80)

        print(
            f"{chunk_size}-WORD CHUNK RESULT"
        )

        print("=" * 80)

        print(
            f"Average Precision@3: "
            f"{average_precision:.4f}"
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL CHUNK-SIZE COMPARISON")
    print("=" * 80)

    print(
        "\nChunk Size | Chunks | Average Precision@3"
    )

    print(
        "-" * 55
    )

    for chunk_size in CHUNK_SIZES:

        data = labeled_results[
            str(chunk_size)
        ]

        print(
            f"{chunk_size:>10} | "
            f"{data['number_of_chunks']:>6} | "
            f"{data['average_precision_at_3']:.4f}"
        )

    # ========================================================
    # BEST CHUNK SIZE
    # ========================================================

    best_size = max(
        CHUNK_SIZES,
        key=lambda size:
            labeled_results[
                str(size)
            ]["average_precision_at_3"]
    )

    best_score = labeled_results[
        str(best_size)
    ]["average_precision_at_3"]

    print("\n")
    print("=" * 80)
    print("BEST CHUNK SIZE")
    print("=" * 80)

    print(
        f"Chunk size        : {best_size} words"
    )

    print(
        f"Average Precision@3: {best_score:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            labeled_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n")
    print(
        "Labeled results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()