from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_hr_questions(resume_text, api_key):
    prompt = PromptTemplate(
        input_variables=["resume"],
        template="""
You are an expert HR professional. Based on the resume below, generate 5 unique HR interview questions
that assess behavioral, situational, and communication skills. Avoid technical or domain-specific questions.

Resume:
{resume}

Questions:"""
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4, google_api_key=api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(resume=resume_text)
    return result.strip().split("\n")[:5]
