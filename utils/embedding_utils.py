import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]


def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]
