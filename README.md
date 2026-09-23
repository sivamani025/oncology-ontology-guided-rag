# Oncology Ontology-Guided RAG System

An oncology-focused Retrieval-Augmented Generation (RAG) system that combines semantic vector retrieval, lexical retrieval, ontology-guided retrieval, vector database frameworks, and an open-weight language model.

## Project Overview

This project implements an end-to-end RAG pipeline for retrieving and generating answers from a small oncology document collection.

The system was developed as part of an Information Retrieval project covering:

* Document chunking
* Sentence embeddings
* Manual vector similarity search
* Retrieval evaluation
* Hybrid retrieval
* Ontology-guided retrieval
* FAISS vector search
* ChromaDB vector search
* Retrieval latency benchmarking
* RAG-based answer generation

## System Architecture

```text
Oncology Documents
       |
       v
Document Preprocessing
       |
       v
Chunking
(500 words, 100-word overlap)
       |
       v
MiniLM Embeddings
(all-MiniLM-L6-v2)
       |
       +--------------------+
       |                    |
       v                    v
Semantic Retrieval     Hybrid Retrieval
       |                    |
       |              TF-IDF + Semantic
       |                    |
       +---------+----------+
                 |
                 v
        Ontology-Guided Retrieval
                 |
                 v
          Top-3 Context Chunks
                 |
                 v
       SmolLM2-360M-Instruct
                 |
                 v
        Generated Answer
                 |
                 v
        Source Chunk Citations
```

## Dataset

The project uses a small oncology document collection containing 11 cancer-related documents.

The documents cover topics including:

* Cancer management
* Cancer biology
* Cancer diagnosis
* Cervical cancer
* Chemotherapy
* Radiation therapy
* Melanoma
* Osteosarcoma
* Stomach cancer
* Cancer pathology
* General cancer information

The source PDFs are not included in the GitHub repository.

## Technologies

* Python
* NumPy
* Sentence Transformers
* `all-MiniLM-L6-v2`
* Transformers
* SmolLM2-360M-Instruct
* FAISS
* ChromaDB
* TF-IDF
* JSON

## Project Structure

```text
oncology-ontology-guided-rag/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── preprocess_oncology.py
│   ├── generate_embeddings.py
│   ├── manual_vector_search.py
│   ├── chunk_size_experiment.py
│   ├── chunk_size_embeddings.py
│   ├── controlled_chunk_experiment.py
│   ├── controlled_chunk_embeddings.py
│   ├── controlled_retrieval_evaluation.py
│   ├── retrieval_evaluation.py
│   ├── label_and_calculate_precision.py
│   ├── hybrid_retrieval_evaluation.py
│   ├── label_hybrid_precision.py
│   ├── ontology.py
│   ├── ontology_guided_retrieval.py
│   ├── label_ontology_precision.py
│   ├── faiss_retrieval.py
│   ├── chroma_retrieval.py
│   ├── database_comparison.py
│   ├── database_scalability.py
│   ├── rag_pipeline.py
│   └── rag_evaluation.py
│
├── results/
│   ├── controlled_precision_results.json
│   ├── database_comparison_results.json
│   ├── database_scalability_results.json
│   ├── ontology_guided_results.json
│   ├── ontology_precision_results.json
│   └── rag_evaluation_results.json
│
├── data/
└── docs/
```

## Document Chunking

Different chunk sizes were evaluated using controlled 20% overlap.

| Chunk Size | Overlap | Chunks |
| ---------: | ------: | -----: |
|        100 |      20 |  1,911 |
|        200 |      40 |  1,023 |
|        300 |      60 |    747 |
|        500 |     100 |    517 |

The controlled retrieval evaluation used 15 oncology questions and `top-k = 3`.

### Precision@3

| Chunk Size | Precision@3 |
| ---------: | ----------: |
|  100 words |      0.6889 |
|  200 words |      0.6000 |
|  300 words |      0.8667 |
|  500 words |      0.9556 |

The 500-word chunk configuration with 100-word overlap was used for the later retrieval and database experiments.

## Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting embedding dimension is:

```text
384
```

The embeddings are stored as NumPy arrays during experimentation.

## Manual Vector Search

Semantic retrieval is implemented using NumPy matrix operations.

For a query vector `q` and document embedding `d`, cosine similarity is calculated as:

```text
cosine_similarity(q,d) =
(q · d) / (||q|| ||d||)
```

The system retrieves the top 3 chunks according to cosine similarity.

## Hybrid Retrieval

Hybrid retrieval combines semantic similarity with lexical similarity.

The implemented scoring approach uses:

```text
Hybrid Score =
0.7 × Semantic Score +
0.3 × Lexical Score
```

The lexical component uses TF-IDF representations.

## Ontology-Guided Retrieval

An oncology ontology was created to represent important concepts and relationships.

The ontology includes concepts such as:

* Cancer
* Melanoma
* Stomach Cancer
* Cervical Cancer
* Osteosarcoma
* Symptom
* Risk Factor
* Biomarker
* Diagnosis
* Biopsy
* Imaging
* Treatment
* Surgery
* Chemotherapy
* Radiation Therapy
* Immunotherapy
* Targeted Therapy
* Supportive Care
* Palliative Care
* Cancer Stage
* Side Effect

Ontology-guided retrieval combines semantic, lexical, and ontology-related scores.

The evaluated ontology-guided retrieval achieved:

```text
Precision@3 = 0.6222
```

## Vector Database Comparison

The project compares three retrieval approaches:

1. NumPy matrix search
2. FAISS
3. ChromaDB

The benchmark was performed using the 500-word chunk configuration.

### Retrieval Latency

Latency is reported in milliseconds.

| Corpus Size |     NumPy |     FAISS |  ChromaDB |
| ----------: | --------: | --------: | --------: |
|         100 | 0.0067 ms | 0.0059 ms | 0.3687 ms |
|       1,000 | 0.0273 ms | 0.0248 ms | 0.3294 ms |
|       5,000 | 0.1917 ms | 0.1455 ms | 0.4366 ms |
|      10,000 | 0.2931 ms | 0.3001 ms | 0.4461 ms |

The 10,000-chunk experiment was included as an additional scalability observation.

## RAG Generation

Retrieved contexts are passed to:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

The generation pipeline follows:

```text
User Question
      |
      v
MiniLM Query Embedding
      |
      v
Cosine Similarity Search
      |
      v
Top-3 Retrieved Chunks
      |
      v
Context + Question
      |
      v
SmolLM2
      |
      v
Generated Answer
      |
      v
Source Chunk Citations
```

The generated response includes source document and chunk identifiers.

## RAG Evaluation

The final RAG pipeline was evaluated using 10 oncology questions covering:

1. Cancer symptoms and signs
2. Cancer risk factors
3. Cancer diagnosis
4. Biopsy
5. Cancer treatment
6. Chemotherapy
7. Radiation therapy
8. Immunotherapy
9. Treatment side effects
10. Cancer staging

The evaluation results are stored in:

```text
results/rag_evaluation_results.json
```

The generated answers demonstrate that retrieved oncology context can be passed to a small open-weight language model to produce natural-language responses with source attribution.

## Example

### Question

```text
What is radiation therapy used for?
```

### Retrieved Sources

```text
cancer-tata-memorial-centre
cancer-management-the-international-agency-for-research-on-cancer-3673
radiation-therapy-and-you-national-cancer-institute-3665
```

### Generated Answer

```text
Radiation therapy is used for treating cancer by using high-energy
rays to kill cancer cells and shrink tumors.
```

## Installation

Create a Python environment and install the required dependencies.

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Preprocess documents

```bash
python src/preprocess_oncology.py
```

### 2. Generate embeddings

```bash
python src/generate_embeddings.py
```

### 3. Run manual vector search

```bash
python src/manual_vector_search.py
```

### 4. Run retrieval experiments

```bash
python src/controlled_retrieval_evaluation.py
```

### 5. Run ontology-guided retrieval

```bash
python src/ontology_guided_retrieval.py
```

### 6. Run FAISS retrieval

```bash
python src/faiss_retrieval.py
```

### 7. Run ChromaDB retrieval

```bash
python src/chroma_retrieval.py
```

### 8. Run database benchmarks

```bash
python src/database_comparison.py
python src/database_scalability.py
```

### 9. Run RAG generation

```bash
python src/rag_pipeline.py
```

### 10. Run RAG evaluation

```bash
python src/rag_evaluation.py
```

## Results Summary

The experiments demonstrate the complete retrieval and generation workflow.

The controlled chunking experiment showed differences in retrieval quality across chunk sizes, with the 500-word configuration achieving a Precision@3 of 0.9556.

The database benchmark demonstrated that direct NumPy search, FAISS, and ChromaDB can all be used for vector retrieval, with different observed latency characteristics across corpus sizes.

The final RAG experiment connected the retrieval system to SmolLM2-360M-Instruct and generated answers for 10 oncology questions using retrieved document context.

## Limitations

* The dataset is relatively small.
* The source collection contains only 11 documents.
* The evaluation uses a limited set of oncology questions.
* Generated answers depend on the retrieved context and the capabilities of the small language model.
* Some generated answers may contain repetition or incomplete sentences.
* Page-level citation metadata was not preserved in the current chunking pipeline, so source attribution currently uses document and chunk identifiers.
* The system is a research/educational prototype and is not intended to provide clinical decision support.

## Reproducibility

The repository contains the source code and experimental result files required to understand and reproduce the retrieval experiments.

The original oncology PDFs, generated embeddings, intermediate chunk files, and local ChromaDB files are excluded from version control.

This project is intended for a
