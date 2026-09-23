import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = os.path.join(
    "processed_data",
    "oncology_chunks.json"
)

OUTPUT_EMBEDDINGS = os.path.join(
    "processed_data",
    "embeddings.npy"
)

OUTPUT_METADATA = os.path.join(
    "processed_data",
    "embedding_metadata.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 60)
print("ONCOLOGY EMBEDDING GENERATION")
print("=" * 60)

print("\nLoading chunk data...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Number of chunks loaded: {len(chunks)}")


# ============================================================
# VALIDATE CHUNK DATA
# ============================================================

if len(chunks) == 0:
    raise ValueError("The chunk file is empty.")

texts = []

for chunk in chunks:

    if "text" not in chunk:
        raise KeyError(
            f"Chunk {chunk.get('chunk_id', 'UNKNOWN')} "
            "does not contain a 'text' field."
        )

    text = chunk["text"].strip()

    if not text:
        raise ValueError(
            f"Chunk {chunk.get('chunk_id', 'UNKNOWN')} "
            "contains empty text."
        )

    texts.append(text)


print("Chunk validation completed.")


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("\nLoading embedding model...")

print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")
print(f"Batch size: {BATCH_SIZE}")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False
)


# ============================================================
# CONVERT TO NUMPY ARRAY
# ============================================================

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ============================================================
# VERIFY EMBEDDING SHAPE
# ============================================================

print("\nEmbedding generation completed.")

print(f"Embedding shape : {embeddings.shape}")
print(f"Embedding dtype : {embeddings.dtype}")


# Expected shape for this dataset:
# (798, 384)

expected_rows = len(chunks)
expected_dimensions = 384

if embeddings.shape != (
    expected_rows,
    expected_dimensions
):
    raise ValueError(
        "Unexpected embedding shape.\n"
        f"Expected: ({expected_rows}, {expected_dimensions})\n"
        f"Received: {embeddings.shape}"
    )

print(
    f"Shape verification PASSED: "
    f"({expected_rows}, {expected_dimensions})"
)


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if np.isnan(embeddings).any():
    raise ValueError(
        "Embeddings contain NaN values."
    )

if np.isinf(embeddings).any():
    raise ValueError(
        "Embeddings contain infinite values."
    )

print("NaN / infinity check PASSED.")


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    OUTPUT_EMBEDDINGS,
    embeddings
)

print(
    f"\nEmbeddings saved to:\n"
    f"{OUTPUT_EMBEDDINGS}"
)


# ============================================================
# SAVE EMBEDDING METADATA
# ============================================================

metadata = []

for index, chunk in enumerate(chunks):

    metadata.append({
        "embedding_index": index,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "document_title": chunk["document_title"],
        "source_file": chunk["source_file"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "chunk_index": chunk["chunk_index"]
    })


with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Metadata saved to:\n"
    f"{OUTPUT_METADATA}"
)


# ============================================================
# VERIFY SAVED EMBEDDINGS
# ============================================================

print("\nVerifying saved embeddings...")

loaded_embeddings = np.load(
    OUTPUT_EMBEDDINGS
)

print(
    f"Loaded embedding shape: "
    f"{loaded_embeddings.shape}"
)

if np.array_equal(
    embeddings,
    loaded_embeddings
):
    print("Saved embedding verification PASSED.")
else:
    raise ValueError(
        "Saved embeddings do not match "
        "the generated embeddings."
    )


# ============================================================
# DISPLAY SAMPLE INFORMATION
# ============================================================

print("\nSample embedding information:")

print(
    f"First chunk ID       : "
    f"{chunks[0]['chunk_id']}"
)

print(
    f"First document       : "
    f"{chunks[0]['document_title']}"
)

print(
    f"First embedding size : "
    f"{len(embeddings[0])}"
)

print(
    f"First embedding      :\n"
    f"{embeddings[0]}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 60)

print(f"Model              : {MODEL_NAME}")
print(f"Number of chunks   : {len(chunks)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Embedding matrix   : {embeddings.shape}")
print(f"Data type          : {embeddings.dtype}")

print("\nGenerated files:")

print(
    f"1. {OUTPUT_EMBEDDINGS}"
)

print(
    f"2. {OUTPUT_METADATA}"
)

print("=" * 60)