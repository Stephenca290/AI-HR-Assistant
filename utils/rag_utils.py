from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store(docs, names, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(splitter.split_text(doc))
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.from_texts(all_chunks, embedding=embeddings)
    db.save_local("faiss_index")

def query_resume_similarity(jd_text, resumes_texts, resume_names, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    jd_embedding = embeddings.embed_query(jd_text)
    scores = []

    for name, resume_text in zip(resume_names, resumes_texts):
        resume_embedding = embeddings.embed_query(resume_text)
        similarity = cosine_similarity(jd_embedding, resume_embedding)
        scores.append((name, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def cosine_similarity(vec1, vec2):
    import numpy as np
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
