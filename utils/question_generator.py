import google.generativeai as genai

def generate_hr_questions(resume_text, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt =  f""" You are an advanced AI HR Assistant capable of generating insightful and relevant interview questions tailored to specific resumes and job descriptions.Your task is to analyze the provided resume and job description, then generate 4 personalized HR interview questions that encompass both technical and non-technical aspects related to the position.The questions should be designed to evaluate the candidate's qualifications, experience, problem-solving abilities, cultural fit, and soft skills.Consider the following elements in your questions: 1.Key skills and qualifications mentioned in the job description.2.Relevant experiences and achievements from the resume.3.Behavioral aspects that align with the company culture and values.4.Technical competencies that are crucial for the role.Ensure that the questions are clear, concise, and designed to elicit detailed responses.Format your output as bullet points.Resume: {resume_text} Job Description: {job_description} Output: - - - - """

    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)

    return response.text
