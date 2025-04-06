import streamlit as st
 from utils.pdf_parser import extract_text_from_pdf
 from utils.rag_utils import create_vector_store, query_resume_similarity
 from utils.question_generator import generate_hr_questions
 import tempfile
with st.sidebar:
    st.title("Menu:")
    
    # Job Description Input
    job_desc = st.text_area("Paste the Job Description", height=200, key="job_desc_input")

    # Resume PDF Upload
    pdf_docs = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, key="resume_uploader")

    if st.button("Submit", key="process_button") and job_desc.strip():  # Ensure JD is provided
        with st.spinner("Processing..."):
            resume_names = [pdf.name for pdf in pdf_docs]
            resumes_texts = [get_pdf_text([pdf]) for pdf in pdf_docs]

            # Rank resumes (if that function is in your code)
            ranking = rank_resumes_by_similarity(resumes_texts, job_desc, api_key)
            st.success("Resumes ranked based on job description.")

            # Show Rankings
            st.subheader("📊 Resume Rankings")
            for name, score in ranking:
                st.write(f"{name}: {score:.2f}")

            # Generate HR Questions
            st.subheader("🎯 HR Interview Questions")
            for name, resume_text in zip(resume_names, resumes_texts):
                questions = generate_hr_questions(resume_text, job_desc, api_key)
                st.markdown(f"**{name}**")
                for q in questions:
                    st.write(f"- {q}")
