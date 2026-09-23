import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from ontology import find_concepts, expand_concepts


CHUNKS = "processed_data/chunks_500_standard.json"
EMBEDDINGS = "processed_data/embeddings_500_standard.npy"
OUTPUT = "processed_data/ontology_guided_results.json"

TOP_K = 3
ALPHA = 0.5
BETA = 0.3
GAMMA = 0.2


def cosine(q, x):
    return np.dot(x, q) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(q)
    )


def main():

    with open(CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS)

    texts = [x["text"] for x in chunks]

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf = vectorizer.fit_transform(texts)

    questions = [
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

    results = {}

    for question in questions:

        concepts = find_concepts(question)
        concepts = expand_concepts(concepts)

        q_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )[0]

        semantic = cosine(
            q_embedding,
            embeddings
        )

        q_tfidf = vectorizer.transform([question])
        lexical = (tfidf @ q_tfidf.T).toarray().flatten()

        ontology = np.array([
            sum(
                1 for concept in concepts
                if concept.lower() in text.lower()
            )
            for text in texts
        ])

        if ontology.max() > 0:
            ontology = ontology / ontology.max()

        semantic = (
            semantic - semantic.min()
        ) / (
            semantic.max() - semantic.min() + 1e-9
        )

        if lexical.max() > 0:
            lexical = (
                lexical - lexical.min()
            ) / (
                lexical.max() - lexical.min() + 1e-9
            )

        final_score = (
            ALPHA * semantic +
            BETA * lexical +
            GAMMA * ontology
        )

        indices = np.argsort(
            final_score
        )[::-1][:TOP_K]

        results[question] = {
            "concepts": concepts,
            "results": []
        }

        for rank, i in enumerate(indices, 1):

            results[question]["results"].append({
                "rank": rank,
                "chunk_id": chunks[i]["chunk_id"],
                "document_title": chunks[i]["document_title"],
                "page": chunks[i]["page_start"],
                "semantic_score": float(semantic[i]),
                "lexical_score": float(lexical[i]),
                "ontology_score": float(ontology[i]),
                "final_score": float(final_score[i]),
                "text": chunks[i]["text"]
            })

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("Ontology-guided retrieval complete.")
    print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()