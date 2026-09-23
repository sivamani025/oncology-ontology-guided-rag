import json
import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "processed_data/extracted_documents.json"
OUTPUT_DIR = "processed_data"

CHUNK_SIZES = [100, 300, 500]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Fix words broken across lines
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace newlines with spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(words, chunk_size, overlap):

    chunks = []

    step = chunk_size - overlap

    start = 0
    chunk_index = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk_words = words[start:end]

        if len(chunk_words) == 0:
            break

        chunks.append({
            "chunk_index": chunk_index,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words)
        })

        chunk_index += 1

        if end == len(words):
            break

        start += step

    return chunks


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def generate_chunks(documents, chunk_size):

    overlap = int(chunk_size * 0.20)

    all_chunks = []

    global_chunk_index = 0

    for document in documents:

        document_id = document["document_id"]
        title = document.get("title", document_id)
        filename = document.get("filename", "")

        pages = document.get("pages", [])

        # ----------------------------------------------------
        # Process each page
        # ----------------------------------------------------

        for page in pages:

            page_number = page.get("page")

            text = clean_text(page.get("text", ""))

            if not text:
                continue

            words = text.split()

            page_chunks = create_chunks(
                words,
                chunk_size,
                overlap
            )

            for chunk in page_chunks:

                record = {
                    "chunk_id":
                        f"{document_id}_SIZE{chunk_size}_CHUNK_{global_chunk_index:04d}",

                    "document_id":
                        document_id,

                    "document_title":
                        title,

                    "source_file":
                        filename,

                    "page_start":
                        page_number,

                    "page_end":
                        page_number,

                    "chunk_index":
                        global_chunk_index,

                    "chunk_size_words":
                        chunk_size,

                    "overlap_words":
                        overlap,

                    "actual_word_count":
                        chunk["word_count"],

                    "text":
                        chunk["text"]
                }

                all_chunks.append(record)

                global_chunk_index += 1

    return all_chunks


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("ONCOLOGY CHUNK SIZE EXPERIMENT")
print("=" * 70)


# Load extracted documents
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


print("\nDocuments loaded :", len(documents))


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GENERATE EACH CHUNK SIZE
# ============================================================

for chunk_size in CHUNK_SIZES:

    print("\n" + "-" * 70)
    print("Generating", chunk_size, "word chunks...")
    print("-" * 70)

    chunks = generate_chunks(
        documents,
        chunk_size
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"chunks_{chunk_size}.json"
    )

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

    print("Chunk size :", chunk_size)
    print("Overlap    :", int(chunk_size * 0.20))
    print("Chunks     :", len(chunks))
    print("Saved to   :", output_file)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHUNK SIZE EXPERIMENT COMPLETE")
print("=" * 70)

print("""
Generated datasets:

100 words  → processed_data/chunks_100.json
300 words  → processed_data/chunks_300.json
500 words  → processed_data/chunks_500.json
""")