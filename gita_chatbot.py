import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langserve import add_routes
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.tools import Tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
try:
    loader = TextLoader("/Users/09hritik/Gita Chatbot/rag/bhagavad_gita.txt")
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(text_documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise RuntimeError("Failed to load Bhagavad Gita text into ChromaDB")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """You are a divine guide inspired by Lord Krishna's wisdom from the Bhagavad Gita. 
    Use the given context to answer queries in about 200 words concise with compassion, wisdom, and guidance. 

    <context> {context} </context>
    Question: {input}
    
    Answer as if Krishna is speaking to Arjuna.
    
    {agent_scratchpad}
    """
)

# Initialize OpenAI LLM
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Wikipedia API wrapper
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# Arxiv API wrapper
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

# Define Bhagavad Gita Retrieval Tool
def retrieval_tool(query: str) -> str:
    """Retrieves relevant Bhagavad Gita passages based on the query."""
    if db:
        docs = db.similarity_search(query, k=1)  # Reduce the number of retrieved documents to 1
        return "\n\n".join([doc.page_content[:500] for doc in docs])  # Truncate long documents to 500 characters
    return "No Bhagavad Gita knowledge available."

retrieval_tool_openai = Tool(
    name="BhagavadGitaRetrieval",
    func=retrieval_tool,
    description="Retrieves Bhagavad Gita knowledge based on the input query."
)

# Define LangChain Agent
tools = [wiki_tool, arxiv_tool, retrieval_tool_openai]
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

# Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# API Endpoint for Chatbot
@app.post("/answer")
async def answer_endpoint(request: QueryRequest):
    query = request.input

    try:
        context = retrieval_tool(query)  # Get Bhagavad Gita context
        logger.info(f"Context: {context}")
        response = agent_executor.invoke({"input": query, "context": context, "agent_scratchpad": ""})
        logger.info(f"Response: {response}")
        return {"answer": response["output"]}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)