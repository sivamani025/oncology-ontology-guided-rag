import os
import re
import json
import zipfile
from pypdf import PdfReader


# ============================================================
# 1. CONFIGURATION
# ============================================================

ZIP_FILE = "Cancer_data.zip"

EXTRACT_DIR = "oncology_data"
OUTPUT_DIR = "processed_data"

CHUNK_SIZE = 200
OVERLAP = 20


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. EXTRACT ZIP FILE
# ============================================================

print("Extracting ZIP file...")

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extraction completed.")


# ============================================================
# 4. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words broken by hyphen at line endings
    text = re.sub(r"-\n(?=\w)", "", text)

    # Replace remaining line breaks with spaces
    text = re.sub(r"\n+", " ", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# ============================================================
# 5. EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text()
        except Exception:
            text = ""

        text = clean_text(text)

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# ============================================================
# 6. GET PDF FILES
# ============================================================

pdf_files = []

for root, directories, files in os.walk(EXTRACT_DIR):

    for file in files:

        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(root, file)
            pdf_files.append(pdf_path)


pdf_files.sort()

print("\nPDF files found:", len(pdf_files))

for pdf in pdf_files:
    print("-", os.path.basename(pdf))


# ============================================================
# 7. EXTRACT ALL DOCUMENTS
# ============================================================

documents = []

print("\nExtracting PDF text...\n")

for document_id, pdf_path in enumerate(pdf_files, start=1):

    filename = os.path.basename(pdf_path)

    print(
        f"[{document_id}/{len(pdf_files)}] "
        f"Processing: {filename}"
    )

    pages = extract_pdf(pdf_path)

    document = {
        "document_id": f"DOC_{document_id:03d}",
        "filename": filename,
        "title": os.path.splitext(filename)[0],
        "pages": pages
    }

    documents.append(document)


# ============================================================
# 8. SAVE EXTRACTED DOCUMENT TEXT
# ============================================================

documents_file = os.path.join(
    OUTPUT_DIR,
    "extracted_documents.json"
)

with open(
    documents_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nSaved:", documents_file)


# ============================================================
# 9. CHUNKING FUNCTION
# ============================================================

def create_chunks(document, chunk_size=200, overlap=20):

    chunks = []

    # Number of new words added after each chunk
    step = chunk_size - overlap

    # --------------------------------------------------------
    # Combine page text but keep page boundaries
    # --------------------------------------------------------

    words_with_pages = []

    for page_data in document["pages"]:

        page_number = page_data["page"]

        words = page_data["text"].split()

        for word in words:
            words_with_pages.append(
                (word, page_number)
            )

    # --------------------------------------------------------
    # Generate overlapping chunks
    # --------------------------------------------------------

    total_words = len(words_with_pages)

    chunk_index = 0

    start = 0

    while start < total_words:

        end = min(
            start + chunk_size,
            total_words
        )

        selected_words = words_with_pages[start:end]

        if not selected_words:
            break

        text = " ".join(
            word for word, page in selected_words
        )

        page_numbers = [
            page for word, page in selected_words
        ]

        chunk = {

            "chunk_id":
                f"{document['document_id']}_"
                f"CHUNK_{chunk_index:04d}",

            "document_id":
                document["document_id"],

            "document_title":
                document["title"],

            "source_file":
                document["filename"],

            "page_start":
                min(page_numbers),

            "page_end":
                max(page_numbers),

            "chunk_index":
                chunk_index,

            "chunk_size_words":
                len(selected_words),

            "overlap_words":
                overlap if start > 0 else 0,

            "text":
                text,

            "concepts":
                []
        }

        chunks.append(chunk)

        chunk_index += 1

        # Move forward while keeping overlap
        start += step

    return chunks


# ============================================================
# 10. CREATE ALL CHUNKS
# ============================================================

all_chunks = []

print("\nCreating chunks...\n")

for document in documents:

    document_chunks = create_chunks(
        document,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    all_chunks.extend(document_chunks)

    print(
        document["document_id"],
        "→",
        len(document_chunks),
        "chunks"
    )


# ============================================================
# 11. SAVE CHUNKS
# ============================================================

chunks_file = os.path.join(
    OUTPUT_DIR,
    "oncology_chunks.json"
)

with open(
    chunks_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 12. SAVE SIMPLE TEXT DATASET
# ============================================================

text_file = os.path.join(
    OUTPUT_DIR,
    "oncology_chunks.txt"
)

with open(
    text_file,
    "w",
    encoding="utf-8"
) as f:

    for chunk in all_chunks:

        f.write(
            f"CHUNK_ID: {chunk['chunk_id']}\n"
        )

        f.write(
            f"SOURCE: {chunk['source_file']}\n"
        )

        f.write(
            f"PAGES: "
            f"{chunk['page_start']}-"
            f"{chunk['page_end']}\n"
        )

        f.write(
            f"{chunk['text']}\n"
        )

        f.write(
            "\n" + "=" * 80 + "\n\n"
        )


# ============================================================
# 13. DATASET STATISTICS
# ============================================================

total_documents = len(documents)
total_chunks = len(all_chunks)

word_counts = [
    len(chunk["text"].split())
    for chunk in all_chunks
]

average_words = (
    sum(word_counts) / len(word_counts)
    if word_counts
    else 0
)

print("\n")
print("=" * 60)
print("ONCOLOGY DATASET PREPROCESSING COMPLETE")
print("=" * 60)

print(
    "Number of documents :",
    total_documents
)

print(
    "Number of chunks    :",
    total_chunks
)

print(
    "Chunk size          :",
    CHUNK_SIZE,
    "words"
)

print(
    "Overlap             :",
    OVERLAP,
    "words"
)

print(
    "Average chunk size  :",
    round(average_words, 2),
    "words"
)

print(
    "JSON output         :",
    chunks_file
)

print(
    "Text output         :",
    text_file
)

print("=" * 60)