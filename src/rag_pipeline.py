import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM

CHUNKS="processed_data/chunks_500_standard.json"
EMBEDDINGS="processed_data/embeddings_500_standard.npy"
MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"

question=input("\nQuestion: ")

with open(CHUNKS,encoding="utf-8") as f:
    chunks=json.load(f)

embeddings=np.load(EMBEDDINGS).astype("float32")
embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

q=embedder.encode(question,convert_to_numpy=True).astype("float32")
scores=embeddings@q/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(q)+1e-10)
indices=np.argsort(scores)[::-1][:3]

context=""
for i in indices:
    x=chunks[i]
    context+=f"\n[{x['document_title']}, Chunk {x['chunk_id']}]\n{x['text']}\n"

tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL)

messages=[
    {
        "role":"system",
        "content":"Answer using only the provided context. Give a concise answer and cite the sources."
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
output=model.generate(**inputs,max_new_tokens=350,do_sample=False)

answer=tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nANSWER:\n")
print(answer)

print("\nSOURCES:")
for i in indices:
    x=chunks[i]
    print(f"- {x['document_title']}, Chunk {x['chunk_id']}")