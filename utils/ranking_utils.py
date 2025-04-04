from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_resumes(query_embedding, resume_embeddings):
    sims = cosine_similarity([query_embedding], resume_embeddings)[0]
    return sims.argsort()[::-1], sims
