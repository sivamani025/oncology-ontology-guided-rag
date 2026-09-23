import json
import os

INPUT = "processed_data/ontology_guided_results.json"
OUTPUT = "processed_data/ontology_precision_results.json"
TOP_K = 3


def label_result(result):
    print("\nRank:", result["rank"])
    print("Chunk:", result["chunk_id"])
    print("Document:", result["document_title"])
    print("Page:", result["page"])
    print("Score:", round(result["final_score"], 4))
    print("\n", result["text"])

    while True:
        x = input("\nRelevant? [1=Yes, 0=No]: ").strip()
        if x in ["0", "1"]:
            return int(x)
        print("Enter 1 or 0.")


def main():

    if not os.path.exists(INPUT):
        print("File not found:", INPUT)
        return

    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    precisions = []

    for n, (question, info) in enumerate(data.items(), 1):

        print("\n" + "=" * 80)
        print(f"QUESTION {n}/{len(data)}")
        print("=" * 80)
        print(question)

        labels = []

        for result in info["results"][:TOP_K]:
            labels.append(label_result(result))

        relevant = sum(labels)
        precision = relevant / TOP_K
        precisions.append(precision)

        results[question] = {
            "concepts": info.get("concepts", []),
            "labels": labels,
            "relevant": relevant,
            "precision_at_3": precision
        }

        print(
            f"\nPrecision@3: {precision:.4f}"
        )

    average = sum(precisions) / len(precisions)

    output = {
        "method": "Ontology-Guided Retrieval",
        "top_k": TOP_K,
        "number_of_questions": len(data),
        "average_precision_at_3": average,
        "per_question": results
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ONTOLOGY EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Average Precision@3: {average:.4f}")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()