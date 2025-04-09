from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
generic_template = "Translate the following into {language}:"

prompt = ChatPromptTemplate.from_messages(

    [("system",generic_template),("user","{text}")]
)

parser = StrOutputParser()

chain = prompt | model | parser


##APP

app = FastAPI(title = "langchain_server",version = "1.0",description = "Simple API server runnable interface")

## Adding Chain routes
add_routes(

    app,
    chain,
    path = "/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host = "localhost" , port = 8000)   # write /docs