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

BATCH_SIZE = 32


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(chunk_size):

    input_file = os.path.join(
        DATA_DIR,
        f"chunks_{chunk_size}_standard.json"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Chunk file not found: {input_file}"
        )

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    return chunks, input_file


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(model, texts):

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    return embeddings.astype(np.float32)


# ============================================================
# SAVE EMBEDDING METADATA
# ============================================================

def create_metadata(chunks):

    metadata = []

    for i, chunk in enumerate(chunks):

        metadata.append({
            "embedding_index": i,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "source_file": chunk["source_file"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_index": chunk["chunk_index"],
            "chunk_size_words": chunk["chunk_size_words"],
            "overlap_words": chunk["overlap_words"]
        })

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONTROLLED CHUNK EMBEDDING GENERATION")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Process each chunk size
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        print("\n" + "-" * 80)
        print(f"PROCESSING {chunk_size}-WORD CHUNKS")
        print("-" * 80)

        # Load chunks
        chunks, input_file = load_chunks(chunk_size)

        print(f"Input file   : {input_file}")
        print(f"Number chunks: {len(chunks)}")

        # Extract text
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = generate_embeddings(
            model,
            texts
        )

        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        expected_shape = (
            len(chunks),
            384
        )

        if embeddings.shape != expected_shape:

            raise ValueError(
                f"Unexpected embedding shape: "
                f"{embeddings.shape}. "
                f"Expected {expected_shape}."
            )

        # Check NaN / Inf
        if np.isnan(embeddings).any():

            raise ValueError(
                "Embedding matrix contains NaN values."
            )

        if np.isinf(embeddings).any():

            raise ValueError(
                "Embedding matrix contains infinite values."
            )

        # ----------------------------------------------------
        # Save embeddings
        # ----------------------------------------------------

        embedding_file = os.path.join(
            DATA_DIR,
            f"embeddings_{chunk_size}_standard.npy"
        )

        np.save(
            embedding_file,
            embeddings
        )

        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        metadata = create_metadata(chunks)

        metadata_file = os.path.join(
            DATA_DIR,
            f"embedding_metadata_{chunk_size}_standard.json"
        )

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

        # ----------------------------------------------------
        # Reload verification
        # ----------------------------------------------------

        reloaded = np.load(
            embedding_file
        )

        if np.array_equal(
            embeddings,
            reloaded
        ):

            verification = "PASSED"

        else:

            verification = "FAILED"

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print(f"Embedding shape : {embeddings.shape}")
        print(f"Data type       : {embeddings.dtype}")
        print(
            f"Saved embeddings: "
            f"{embedding_file}"
        )
        print(
            f"Saved metadata  : "
            f"{metadata_file}"
        )
        print(
            f"Reload verification: "
            f"{verification}"
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONTROLLED EMBEDDING GENERATION COMPLETE")
    print("=" * 80)

    print("\nExpected files:")

    for chunk_size in CHUNK_SIZES:

        print(
            f"  embeddings_{chunk_size}_standard.npy"
        )

        print(
            f"  embedding_metadata_{chunk_size}_standard.json"
        )

    print("\nNext step:")
    print(
        "Run the controlled retrieval evaluation "
        "and calculate Precision@3."
    )


if __name__ == "__main__":
    main()