import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM

CHUNKS="processed_data/chunks_500_standard.json"
EMBEDDINGS="processed_data/embeddings_500_standard.npy"
MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"

questions=[
    "What are the common symptoms and signs of cancer?",
    "What are the major risk factors for cancer?",
    "What methods are used to diagnose cancer?",
    "What is the role of biopsy in cancer diagnosis?",
    "What are the main types of cancer treatment?",
    "What is chemotherapy and how is it used in cancer treatment?",
    "What is radiation therapy used for?",
    "What is immunotherapy in cancer treatment?",
    "What are common side effects of cancer treatment?",
    "Why is cancer staging important?"
]

with open(CHUNKS,encoding="utf-8") as f:
    chunks=json.load(f)

embeddings=np.load(EMBEDDINGS).astype("float32")
embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL)

results=[]

for question in questions:
    q=embedder.encode(question,convert_to_numpy=True).astype("float32")
    scores=embeddings@q/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(q)+1e-10)
    indices=np.argsort(scores)[::-1][:3]

    context=""
    sources=[]

    for i in indices:
        x=chunks[i]
        context+=f"\n[{x['document_title']}, Chunk {x['chunk_id']}]\n{x['text']}\n"
        sources.append({
            "document":x["document_title"],
            "chunk_id":x["chunk_id"]
        })

    messages=[
        {
            "role":"system",
            "content":"Answer the question using only the provided context. Give a short, non-repetitive answer with 5 to 8 key points. Do not repeat any symptom. Stop after the answer is complete. Cite the relevant chunk IDs."
        },
        {
            "role":"user",
            "content":f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    prompt=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs=tokenizer(prompt,return_tensors="pt")

    output=model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        no_repeat_ngram_size=3
    )

    answer=tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    results.append({
        "question":question,
        "answer":answer,
        "sources":sources
    })

    print("\nQUESTION:",question)
    print("ANSWER:",answer)
    print("SOURCES:",sources)

with open("processed_data/rag_evaluation_results.json","w",encoding="utf-8") as f:
    json.dump(results,f,indent=2,ensure_ascii=False)

print("\nSaved to processed_data/rag_evaluation_results.json")