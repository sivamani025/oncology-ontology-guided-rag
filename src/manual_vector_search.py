import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. LOAD DATA
# ============================================================

CHUNKS_FILE = "processed_data/oncology_chunks.json"
EMBEDDINGS_FILE = "processed_data/embeddings.npy"

print("=" * 60)
print("LOADING ONCOLOGY DATA")
print("=" * 60)

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load document embeddings
document_embeddings = np.load(EMBEDDINGS_FILE)

print("Number of chunks     :", len(chunks))
print("Embedding matrix     :", document_embeddings.shape)


# ============================================================
# 2. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading model:", model_name)

model = SentenceTransformer(model_name)


# ============================================================
# 3. COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_embedding, document_embeddings):

    # Dot product between query and every document
    dot_products = np.dot(document_embeddings, query_embedding)

    # Norm of every document embedding
    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    # Norm of query embedding
    query_norm = np.linalg.norm(query_embedding)

    # Cosine similarity
    scores = dot_products / (
        document_norms * query_norm
    )

    return scores


# ============================================================
# 4. RETRIEVE TOP-K DOCUMENTS
# ============================================================

def retrieve(query, top_k=3):

    print("\n" + "=" * 60)
    print("QUERY")
    print("=" * 60)

    print(query)

    # Generate query embedding
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    # Make sure it is float32
    query_embedding = query_embedding.astype(np.float32)

    print("\nQuery embedding shape:", query_embedding.shape)

    # Calculate cosine similarity
    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    # Get indices of highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Display results
    print("\n" + "=" * 60)
    print("TOP", top_k, "RETRIEVED CHUNKS")
    print("=" * 60)

    for rank, index in enumerate(top_indices, start=1):

        chunk = chunks[index]

        print("\n" + "-" * 60)
        print("Rank              :", rank)
        print("Similarity Score  :", round(float(scores[index]), 4))
        print("Chunk ID           :", chunk["chunk_id"])
        print("Document           :", chunk["document_title"])
        print("Page               :", 
              chunk["page_start"],
              "to",
              chunk["page_end"])
        print("Source File        :", chunk["source_file"])

        print("\nRetrieved Text:")
        print(chunk["text"])


# ============================================================
# 5. TEST QUERIES
# ============================================================

queries = [

    "What are the common symptoms and signs of cancer?",

    "What are the side effects of chemotherapy?",

    "How is stomach cancer diagnosed and treated?"
]


# ============================================================
# 6. RUN RETRIEVAL
# ============================================================

for query in queries:
    retrieve(query, top_k=3)