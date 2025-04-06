import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.rag_utils import create_vector_store, query_resume_similarity
from utils.question_generator import generate_hr_questions
import tempfile

st.set_page_config(page_title="AI-Powered HR Assistant", layout="wide")

st.title("🤖 AI-Powered HR Assistant")

st.sidebar.header("Upload Job Description and Resumes")

api_key = st.secrets["api_key"]

job_description_file = st.sidebar.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Analyze"):
    if job_description_file and resume_files:
        with st.spinner("Processing..."):
            jd_text = extract_text_from_pdf(job_description_file)
            resumes_texts = [extract_text_from_pdf(resume) for resume in resume_files]
            resume_names = [resume.name for resume in resume_files]

            # Build vector DB
            create_vector_store([jd_text] + resumes_texts, ["JD"] + resume_names, api_key)

            # Similarity scores
            ranking = query_resume_similarity(jd_text, resumes_texts, resume_names, api_key)

            st.subheader("📊 Resume Ranking")
            for rank, (name, score) in enumerate(ranking, 1):
                st.write(f"{rank}. **{name}** - Similarity Score: `{score:.2f}`")

            st.subheader("🎯 HR Interview Questions")
            for name, resume_text in zip(resume_names, resumes_texts):
                questions = generate_hr_questions(resume_text, jd_text, api_key)
                st.markdown(f"**{name}**")
                for q in questions:
                    st.write(f"- {q}")
    else:
        st.warning("Please upload both job description and resumes.")
