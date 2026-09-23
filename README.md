\# Oncology Ontology-Guided RAG System



An oncology-focused Retrieval-Augmented Generation (RAG) system that combines semantic vector retrieval, lexical retrieval, ontology-guided retrieval, vector database frameworks, and an open-weight language model.



\## Project Overview



This project implements an end-to-end RAG pipeline for retrieving and generating answers from a small oncology document collection.



The system was developed as part of an Information Retrieval project covering:



\- Document chunking

\- Sentence embeddings

\- Manual vector similarity search

\- Retrieval evaluation

\- Hybrid retrieval

\- Ontology-guided retrieval

\- FAISS vector search

\- ChromaDB vector search

\- Retrieval latency benchmarking

\- RAG-based answer generation



\## System Architecture



```text

Oncology Documents

&#x20;      |

&#x20;      v

Document Preprocessing

&#x20;      |

&#x20;      v

Chunking

(500 words, 100-word overlap)

&#x20;      |

&#x20;      v

MiniLM Embeddings

(all-MiniLM-L6-v2)

&#x20;      |

&#x20;      +--------------------+

&#x20;      |                    |

&#x20;      v                    v

Semantic Retrieval     Hybrid Retrieval

&#x20;      |                    |

&#x20;      |              TF-IDF + Semantic

&#x20;      |                    |

&#x20;      +---------+----------+

&#x20;                |

&#x20;                v

&#x20;       Ontology-Guided Retrieval

&#x20;                |

&#x20;                v

&#x20;         Top-3 Context Chunks

&#x20;                |

&#x20;                v

&#x20;      SmolLM2-360M-Instruct

&#x20;                |

&#x20;                v

&#x20;       Generated Answer

&#x20;                |

&#x20;                v

&#x20;       Source Chunk Citations

