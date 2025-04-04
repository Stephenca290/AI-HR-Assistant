import openai

import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_hr_questions(resume_text, job_description):
    prompt = f"""
You are an HR professional. Given the following resume and job description, generate 3 tailored interview questions:

Resume:
{resume_text}

Job Description:
{job_description}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()
