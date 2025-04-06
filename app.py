import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.rag_utils import create_vector_store, query_resume_similarity
from utils.question_generator import generate_hr_questions
import tempfile

st.set_page_config(page_title="AI HR Assistant", layout="wide")
st.title("🤖 AI-Powered HR Assistant")

uploaded_files = st.file_uploader("Upload resumes (PDF only)", type="pdf", accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if st.button("Analyze Resumes"):
    if not uploaded_files or not job_description:
        st.warning("Please upload resumes and enter a job description.")
    else:
        with st.spinner("Processing..."):
            resume_texts = []
            resume_names = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resume_texts.append(text)
                resume_names.append(file.name)

            db, index = create_vector_store(resume_texts, resume_names)
            ranked = query_resume_similarity(db, index, job_description)

            st.subheader("📊 Ranked Resumes & HR Questions")
            for name, content, score in ranked:
                st.markdown(f"### {name} — Similarity Score: {score:.2f}")
                st.markdown(f"**Resume Extract:**\n\n{content[:400]}...")
                questions = generate_hr_questions(content)
                st.markdown("**HR Questions:**")
                for q in questions:
                    st.markdown(f"- {q}")
