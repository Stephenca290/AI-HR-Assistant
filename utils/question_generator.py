import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

def generate_hr_questions(job_desc, resume_content):
    prompt = f"""You are an HR interviewer.
Based on the following job description and resume, generate 5 personalized HR interview questions.

Job Description:
{job_desc}

Resume:
{resume_content}

Questions:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip().split("\n")
    except Exception as e:
        print("Error generating questions:", e)
        return ["Could not generate questions due to API error."]
