from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(docs, names):
    embeddings = model.encode(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return {"docs": docs, "names": names, "embeddings": embeddings}, index

def query_resume_similarity(db, index, query, top_k=5):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append((db["names"][idx], db["docs"][idx], 1 / (1 + distances[0][i])))
    return results
