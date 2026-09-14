from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Two Groq models: a fast one for quick tasks, a stronger one for the "essay" route
fast_model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
strong_model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

add_routes(
    app,
    fast_model,
    path="/groq"
)

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1 | strong_model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | fast_model,
    path="/poem"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)