from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(docs, names):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chromadb"))
    try:
        client.delete_collection("resumes")
    except:
        pass
    collection = client.create_collection(name="resumes")

    embeddings = model.encode(docs).tolist()

    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        collection.add(documents=[doc], embeddings=[emb], ids=[str(i)], metadatas=[{"name": names[i]}])

    return collection

def query_resume_similarity(vectordb, job_desc, top_k=5):
    query_embedding = model.encode([job_desc]).tolist()[0]
    results = vectordb.query(query_embeddings=[query_embedding], n_results=top_k)

    ranked = []
    for i in range(len(results["documents"][0])):
        content = results["documents"][0][i]
        score = results["distances"][0][i]
        name = results["metadatas"][0][i]["name"]
        ranked.append({"name": name, "content": content, "score": score})
    return ranked
