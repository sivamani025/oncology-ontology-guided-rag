import json
import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = os.path.join(
    "processed_data",
    "extracted_documents.json"
)

CHUNK_SIZES = [100, 200, 300, 500]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Join words broken by hyphen + newline
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


# ============================================================
# WORD-BASED CHUNKING
# ============================================================

def create_chunks(documents, chunk_size):
    """
    Create chunks independently for each page.

    Overlap = 20% of chunk size.
    """

    overlap = int(chunk_size * 0.20)
    step = chunk_size - overlap

    all_chunks = []

    for doc in documents:

        document_id = doc.get("document_id")
        document_title = doc.get("title", "")
        source_file = doc.get("filename", "")

        chunk_index = 0

        pages = doc.get("pages", [])

        for page in pages:

            page_number = page.get("page_number")
            page_text = page.get("text", "")

            page_text = clean_text(page_text)

            if not page_text:
                continue

            words = page_text.split()

            if not words:
                continue

            start = 0

            while start < len(words):

                end = min(start + chunk_size, len(words))

                chunk_words = words[start:end]

                if not chunk_words:
                    break

                chunk_text = " ".join(chunk_words)

                chunk = {
                    "chunk_id": (
                        f"{document_id}_"
                        f"CHUNK_{chunk_index:04d}"
                    ),
                    "document_id": document_id,
                    "document_title": document_title,
                    "source_file": source_file,
                    "page_start": page_number,
                    "page_end": page_number,
                    "chunk_index": chunk_index,
                    "chunk_size_words": chunk_size,
                    "overlap_words": overlap,
                    "actual_word_count": len(chunk_words),
                    "text": chunk_text
                }

                all_chunks.append(chunk)

                chunk_index += 1

                # Stop when we have reached the end
                if end >= len(words):
                    break

                start += step

    return all_chunks, overlap


# ============================================================
# SAVE JSON
# ============================================================

def save_chunks(chunks, output_file):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONTROLLED CHUNK-SIZE EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # Load extracted documents
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(f"\nERROR: Input file not found:")
        print(INPUT_FILE)
        return

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    print(f"\nDocuments loaded : {len(documents)}")

    # --------------------------------------------------------
    # Generate all chunk sizes
    # --------------------------------------------------------

    results = {}

    for chunk_size in CHUNK_SIZES:

        print("\n" + "-" * 80)
        print(f"CHUNK SIZE: {chunk_size} WORDS")
        print("-" * 80)

        chunks, overlap = create_chunks(
            documents,
            chunk_size
        )

        output_file = os.path.join(
            "processed_data",
            f"chunks_{chunk_size}_standard.json"
        )

        save_chunks(
            chunks,
            output_file
        )

        results[chunk_size] = {
            "chunks": len(chunks),
            "overlap": overlap,
            "file": output_file
        }

        print(f"Chunk size : {chunk_size}")
        print(f"Overlap    : {overlap}")
        print(f"Chunks     : {len(chunks)}")
        print(f"Saved to   : {output_file}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONTROLLED CHUNKING COMPLETE")
    print("=" * 80)

    print("\nSummary:")
    print("-" * 60)

    for chunk_size in CHUNK_SIZES:

        result = results[chunk_size]

        print(
            f"{chunk_size:>4} words | "
            f"overlap = {result['overlap']:>3} | "
            f"chunks = {result['chunks']:>5}"
        )

    print("-" * 60)

    print("\nGenerated files:")

    for chunk_size in CHUNK_SIZES:
        print(
            f"  processed_data/"
            f"chunks_{chunk_size}_standard.json"
        )

    print("\nNext step:")
    print(
        "Generate MiniLM embeddings for these four "
        "standardized chunk sets."
    )


if __name__ == "__main__":
    main()