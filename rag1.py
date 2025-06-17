import bs4
from dotenv import load_dotenv
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from gtts import gtts

load_dotenv()

# Load environment variables
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
    title="Gita Chatbot API",
    description="Chatbot that responds as Lord Krishna using Bhagavad Gita and external knowledge sources.",
    version="1.0"
)

# Request body model
class QueryRequest(BaseModel):
    input: str

# Load Gita text and create vectorstore
loader = TextLoader("/Users/09hritik/Gita Chatbot/rag/bhagavad_gita.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(text_documents)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """You are Lord Krishna — the eternal, all-knowing guide, the inner voice of dharma and truth.

Just as you guided Arjuna on the battlefield of Kurukshetra, now you guide seekers through the challenges of modern life — confusion, ego, fear, desire, and self-doubt.

Your task is to respond to the seeker's message with deep spiritual insight, timeless clarity, and calm assurance.

Given the <context> and the seeker’s question, respond in Krishna’s voice — serene, wise, compassionate, and detached.

🧠 Structure your answer like this:
1. Gently reflect the seeker’s emotional state.
2. Offer divine insight and clarity.
3. Guide them toward their higher Self through dharma, detachment, and faith.
4. End with a relevant shloka from the Bhagavad Gita (with chapter and verse), only if it fits naturally with the message.

Length: ~100-200 words. Be concise but profound.

<context>
{context}
</context>

Seeker’s Question:
{input}

Now, speak as Krishna, guiding the seeker in their modern-day Kurukshetra — toward truth, strength, and self-realization.

    """
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# External knowledge tools
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300))

# Setup LangChain retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Define FastAPI endpoint
@app.post("/answer")
async def answer_endpoint(request: QueryRequest):
    query = request.input
    result = db.similarity_search(query)
    response = retrieval_chain.invoke({"input": query, "context": result})
    return {"answer": response["answer"]}

# Run app
if __name__ == "__main__":
    uvicorn.run("rag1:app", host="127.0.0.1", port=8000, reload=True)
