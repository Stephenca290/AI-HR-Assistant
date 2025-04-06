import streamlit as st
import os
import uuid
from utils import pdf_utils, embedding_utils, chromadb_utils, llm_utils

st.set_page_config(page_title="AI HR Assistant", layout="wide")
st.title("🤖 AI HR Assistant")



uploaded_files = st.file_uploader("Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if st.button("Analyze"):
    if not uploaded_files or not job_description:
        st.warning("Please upload resumes and provide a job description.")
    else:
        job_embed = embedding_utils.get_embedding(job_description)
        results = []

        for file in uploaded_files:
            save_path = f"./data/resumes/{file.name}"
            with open(save_path, "wb") as f:
                f.write(file.read())

            text = pdf_utils.extract_text_from_pdf(save_path)
            resume_id = str(uuid.uuid4())
            embedding = embedding_utils.get_embedding(text)

            chromadb_utils.add_resume_to_db(resume_id, text, embedding)

            similarity = chromadb_utils.query_similar_resumes(job_embed, top_k=5)
            questions = llm_utils.generate_hr_questions(text, job_description)

            results.append((file.name, similarity["distances"][0][0], questions))

        results.sort(key=lambda x: x[1])  # Sort by similarity (lowest distance = highest similarity)

        st.subheader("📄 Ranked Resumes & Suggested Questions")
        for idx, (name, score, questions) in enumerate(results, 1):
            st.markdown(f"### {idx}. {name}")
            st.markdown(f"**Similarity Score**: {round(score, 4)}")
            st.markdown("**Suggested Interview Questions:**")
            st.markdown(questions)
