import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    100: "processed_data/chunks_100.json",
    300: "processed_data/chunks_300.json",
    500: "processed_data/chunks_500.json"
}

OUTPUTS = {
    100: "processed_data/embeddings_100.npy",
    300: "processed_data/embeddings_300.npy",
    500: "processed_data/embeddings_500.npy"
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("CHUNK SIZE EMBEDDING EXPERIMENT")
print("=" * 70)

print("\nLoading model...")
model = SentenceTransformer(MODEL_NAME)

print("Model:", MODEL_NAME)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

for chunk_size in DATASETS:

    input_file = DATASETS[chunk_size]
    output_file = OUTPUTS[chunk_size]

    print("\n" + "-" * 70)
    print("Processing", chunk_size, "word chunks")
    print("-" * 70)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print("Number of chunks:", len(chunks))

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    embeddings = embeddings.astype(np.float32)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    expected_shape = (len(chunks), 384)

    print("Embedding shape:", embeddings.shape)
    print("Expected shape :", expected_shape)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if np.isnan(embeddings).any():
        raise ValueError("NaN values found in embeddings.")

    if np.isinf(embeddings).any():
        raise ValueError("Infinite values found in embeddings.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(output_file, embeddings)

    print("Saved to:", output_file)

    # --------------------------------------------------------
    # Reload verification
    # --------------------------------------------------------

    loaded = np.load(output_file)

    if np.array_equal(embeddings, loaded):
        print("Reload verification: PASSED")
    else:
        print("Reload verification: FAILED")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CHUNK SIZE EMBEDDING EXPERIMENT COMPLETE")
print("=" * 70)

print("""
Generated files:

100 words → processed_data/embeddings_100.npy
300 words → processed_data/embeddings_300.npy
500 words → processed_data/embeddings_500.npy
""")