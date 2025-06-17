# 🕉️ Bhagavad Gita Chatbot

An AI-powered chatbot that delivers **contextual shlokas and divine guidance** from the Bhagavad Gita.  
It uses **Retrieval-Augmented Generation (RAG)** powered by **LangChain**, **ChromaDB**, and **OpenAI GPT-3.5** to simulate **Krishna-like wisdom** in response to user queries.

---

## ⚙️ Tech Stack

- 🧠 **OpenAI GPT-3.5** – For generating wise, context-aware responses  
- 🛠️ **LangChain** – Prompt chaining and tool orchestration  
- 📚 **ChromaDB** – Vector database for storing and retrieving verses  
- 🌐 **FastAPI** – Lightweight backend API framework  
- 🗂️ **Bhagavad Gita Dataset** – Verse-wise structured data (Sanskrit, Translation, Explanation)  
- 🐍 **Python** – Core programming language

---

## 🚀 Key Features

- Accepts **natural language queries or life questions**
- Performs **semantic search** to find relevant shlokas
- Responds with:
  - ✨ Original **Sanskrit verse**
  - 🌍 **Hindi / English translation**
  - 🧘 **Contextual explanation** inspired by Lord Krishna's teachings
- Modular and scalable **FastAPI-based RESTful backend**
- **Ready for integration** with Telegram, web chat UI, or voice interface

---

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/09hritik/Bhagavd-gita-chatbot.git
cd Bhagavd-gita-chatbot

# Install required packages
pip install -r requirements.txt

# Run the application (rag1.py is the FastAPI entrypoint)
uvicorn rag1:app --reload
