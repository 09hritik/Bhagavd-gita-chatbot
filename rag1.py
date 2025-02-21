import bs4
from dotenv import load_dotenv
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langserve import add_routes
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

load_dotenv()

# Ensure the environment variables are set correctly
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
user_agent = os.getenv("USER_AGENT")

if not openai_api_key or not langchain_api_key or not user_agent:
    raise ValueError("Environment variables OPENAI_API_KEY, LANGCHAIN_API_KEY, and USER_AGENT must be set")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["USER_AGENT"] = user_agent

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot using Langchain",
    description="This is a chatbot API using Langchain",
    version="0.1"
)

# Define request model
class QueryRequest(BaseModel):
    input: str

# Load and process documents
loader = TextLoader("/Users/09hritik/Gita Chatbot/rag/bhagavad_gita.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(text_documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a problem solver given the knowledge of the Bhagavad Gita in the context below. Answer the following questions with the help of the context. help people from kalyug and spread divine knowledge. <context> {context} </context> Questions = {input}"
)

# Initialize OpenAI LLM
llm = ChatOpenAI(model='gpt-3.5-turbo')


# Wikipedia API wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki= WikipediaQueryRun(api_wrapper=api_wrapper)

# Arxiv API wrapper
api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv= ArxivQueryRun(api_wrapper=api_wrapper)


# Create chains
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


@app.post("/answer")
async def answer_endpoint(request: QueryRequest):
    query = request.input
    result = db.similarity_search(query)
    response = retrieval_chain.invoke({"input": query, "context": result})
    return {"answer": response['answer']}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)