import google.generativeai as genai

def generate_hr_questions(resume_text, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt =  f""" You are an advanced AI HR Assistant tasked with generating relevant HR interview questions tailored to specific resumes and job descriptions.Your goal is to create a balanced set of technical and non-technical questions that assess both the candidate's skills and cultural fit within the organization.Please follow these guidelines: 1.Analyze the provided resume and job description to identify key skills, experiences, and qualifications.2.Generate 4 concise and focused HR interview questions that cover both technical competencies and soft skills.3.Ensure that each question is relevant to the job role and reflects the candidate's experience as outlined in the resume.4.Format your response as bullet points, avoiding any additional commentary or explanations.Resume: {resume_text} Job Description: {job_description} Output your questions below:"""

    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)

    return response.text
