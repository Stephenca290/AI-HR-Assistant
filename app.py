import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.rag_utils import create_vector_store, query_resume_similarity
from utils.question_generator import generate_hr_questions

st.set_page_config(page_title="AI HR Assistant", layout="wide")
st.title("🤖 AI HR Assistant")

uploaded_files = st.file_uploader("📤 Upload multiple resumes (PDF only)", type="pdf", accept_multiple_files=True)
job_description = st.text_area("📝 Paste the job description here")

if st.button("🚀 Analyze Resumes") and uploaded_files and job_description:
    resume_texts = extract_text_from_pdf(uploaded_files)
    resume_names = [file.name for file in uploaded_files]

    vectordb = create_vector_store(resume_texts, resume_names)
    ranked_resumes = query_resume_similarity(vectordb, job_description, top_k=len(uploaded_files))

    st.subheader("📊 Ranked Resumes & HR Questions")

    for i, res in enumerate(ranked_resumes, 1):
        st.markdown(f"### {i}. {res['name']} (Similarity Score: {res['score']:.4f})")
        with st.expander("💬 Suggested HR Interview Questions"):
            questions = generate_hr_questions(job_description, res["content"])
            for q in questions:
                st.markdown(f"- {q}")
