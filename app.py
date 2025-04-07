import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.rag_utils import create_vector_store, query_resume_similarity
from utils.question_generator import generate_hr_questions
import tempfile

st.set_page_config(page_title="AI-Powered HR Assistant", layout="wide")

st.title("🤖 AI-Powered HR Assistant")

st.sidebar.header("Input Job Description and Upload Resumes")

api_key = st.secrets["api_key"]

# ⬇️ Changed from file upload to text input
job_description_text = st.sidebar.text_area("Paste Job Description Text Here", height=200)

resume_files = st.sidebar.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Analyze"):
    if job_description_text and resume_files:
        with st.spinner("Processing..."):
            jd_text = job_description_text.strip()
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
                if isinstance(questions, str):
                    questions = questions.strip().split("\n")
                questions = [q.strip("-• ") for q in questions if q.strip().endswith("?")]
                st.markdown(f"**{name}**")
                for q in questions:
                    st.write(f"- {q}")
    else:
        st.warning("Please provide both the job description and resumes.")
