from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = [
        ("system", "You are a helpful assistant. Please respond to the user queries")
    ]

def create_prompt():
    return ChatPromptTemplate.from_messages(st.session_state.messages)

st.title('ChatBot Using Llama 3.1')
input_text = st.text_input("Search the topic you want")

llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()

st.session_state.messages.append(("user", f"Question: {input_text}"))

prompt = create_prompt()

chain = prompt | llm | output_parser
response = chain.invoke({'question':input_text})

st.session_state.messages.append(("assistant", response))

st.write(response)


