import google.generativeai as genai

def generate_hr_questions(resume_text, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt = f"""
You are an AI HR Assistant.

Generate 4 personalized HR interview questions based on the following resume and job description.

Resume:
{resume_text}

Job Description:
{job_description}

Only provide the questions as bullet points.

"""

    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)

    return response.text
