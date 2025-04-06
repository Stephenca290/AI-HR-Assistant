from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_hr_questions(resume_text, job_description, api_key):
    prompt_template = """
    You are an AI HR assistant. Your job is to generate 5 personalized HR interview questions 
    based on the applicant's resume and the given job description.

    Resume:
    {resume}

    Job Description:
    {jd}

    Provide the questions in bullet point format.
    """

    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template=prompt_template,
    )

    llm = ChatGoogleGenerativeAI(
        model="models/chat-bison-001",  # Compatible and stable model
        temperature=0.4,
        google_api_key=api_key,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"resume": resume_text, "jd": job_description})
    return output
