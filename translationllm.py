import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192",groq_api_key=groq_api_key)
generic_template = "Please answer the user's question thoroughly in {language}. Craft a response that is descriptive, providing ample detail, context, and background information. At the same time, ensure the answer is clear, easy to understand (using simple language), and structured logically to make the information easy to remember."

prompt = ChatPromptTemplate.from_messages(

    [("system",generic_template),("user","{text}")]
)

parser = StrOutputParser()

chain = prompt | model | parser

st.title("Your language Question Answer AI")

language_text = st.selectbox("Choose your Language you want to translate",("Bengali","English","Hindi","Urdu","French","Spanish"))
input_text=st.text_input("What question you have in mind?")


if input_text and language_text:
    if st.button("Go"):
        response_placeholder = st.empty()  
        response = ""
        for chunk in chain.stream({"language":language_text,"text": input_text}):
            response += chunk
            response_placeholder.markdown(response)
