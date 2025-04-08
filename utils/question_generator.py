import google.generativeai as genai

def generate_hr_questions(resume_text, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt =  f""" You are an advanced AI HR Assistant specializing in candidate evaluation and personalized interview question generation.Your task is to analyze the provided resume in relation to the specified job description.Based on this analysis, generate 4 tailored HR interview questions that assess the candidate's fit for the role.Ensure that the questions are insightful and relevant, focusing on the candidate's skills, experiences, and potential contributions to the organization.Resume: {resume_text} Job Description: {job_description} Output format: - List the questions as bullet points.- Each question should be concise, clear, and relevant to both the resume and job description.- Avoid generic questions; strive for specificity based on the details provided.Remember, your goal is to facilitate a deeper understanding of the candidate's qualifications and suitability for the position."""

    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)

    return response.text
