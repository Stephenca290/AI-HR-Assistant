import google.generativeai as genai

def generate_hr_questions(resume_text, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt = f"""
You are an AI HR Assistant.

Generate 5 personalized HR interview questions based on the following resume and job description.

Resume:
{resume_text}

Job Description:
{job_description}

"""

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return response.text
