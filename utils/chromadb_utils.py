# utils/chromadb_utils.py

import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

import chromadb
from chromadb.config import Settings

client = chromadb.Client()



collection = client.get_or_create_collection("resumes")

def add_resume_to_db(resume_id, text, embedding):
    collection.add(documents=[text], embeddings=[embedding], ids=[resume_id])

def query_similar_resumes(query_embedding, top_k=5):
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
