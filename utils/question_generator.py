import google.generativeai as genai
import os

# Set your Gemini API key in environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-pro")

def generate_hr_questions(resume_text):
    prompt = f"""
    You are an HR assistant. Read this resume and generate 3 smart HR interview questions based on the candidate's skills, experience, and projects.
    
    Resume:
    {resume_text[:1000]}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip().split('\n')
    except Exception as e:
        return [f"Error generating questions: {e}"]
