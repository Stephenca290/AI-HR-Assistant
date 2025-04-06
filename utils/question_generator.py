import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_hr_questions(resume_text, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt = f"""
You are an AI HR Assistant.

Generate 5 personalized HR interview questions based on the following resume and job description.

Resume:
{resume_text}

Job Description:
{job_description}

Return the questions in bullet point format.
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    response = model.generate_content(prompt)

    return response.text
