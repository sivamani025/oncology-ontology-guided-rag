import time,json
import numpy as np
import faiss
import chromadb

EMBEDDINGS="processed_data/embeddings_500_standard.npy"
OUTPUT="processed_data/database_scalability_results.json"
SIZES=[100,1000,5000,10000]
RUNS=20
DIM=384

def numpy_search(data,q):
    t=[]
    for _ in range(RUNS):
        s=time.perf_counter()
        np.argsort(np.dot(data,q))[-3:]
        t.append((time.perf_counter()-s)*1000)
    return np.mean(t)

def faiss_search(data,q):
    data=data.copy()
    faiss.normalize_L2(data)
    index=faiss.IndexFlatIP(DIM)
    index.add(data)
    q=q.reshape(1,-1).copy()
    faiss.normalize_L2(q)
    t=[]
    for _ in range(RUNS):
        s=time.perf_counter()
        index.search(q,3)
        t.append((time.perf_counter()-s)*1000)
    return np.mean(t)

def chroma_search(data,q):
    client=chromadb.Client()
    try:
        client.delete_collection("benchmark")
    except:
        pass

    col=client.create_collection(
        name="benchmark",
        configuration={"hnsw":{"space":"cosine"}}
    )

    for i in range(0,len(data),5000):
        j=min(i+5000,len(data))
        col.add(
            ids=[str(x) for x in range(i,j)],
            embeddings=data[i:j].tolist(),
            documents=["chunk"]*(j-i)
        )

    t=[]
    for _ in range(RUNS):
        s=time.perf_counter()
        col.query(query_embeddings=[q.tolist()],n_results=3)
        t.append((time.perf_counter()-s)*1000)

    return np.mean(t)

def main():
    base=np.load(EMBEDDINGS).astype("float32")
    rng=np.random.default_rng(42)
    results=[]

    for n in SIZES:
        if n<=len(base):
            data=base[:n]
        else:
            extra=rng.normal(size=(n-len(base),DIM)).astype("float32")
            data=np.vstack([base,extra])

        q=rng.normal(size=DIM).astype("float32")
        a=numpy_search(data,q)
        b=faiss_search(data,q)
        c=chroma_search(data,q)

        results.append({
            "chunks":n,
            "NumPy":float(a),
            "FAISS":float(b),
            "ChromaDB":float(c)
        })

        print(f"{n} chunks | NumPy: {a:.4f} ms | FAISS: {b:.4f} ms | ChromaDB: {c:.4f} ms")

    with open(OUTPUT,"w") as f:
        json.dump(results,f,indent=4)

    print("\nResults saved to:",OUTPUT)

if __name__=="__main__":
    main()