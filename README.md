# 🕉️ Bhagavad Gita Chatbot

An AI-powered chatbot that delivers **contextual shlokas and divine guidance** from the Bhagavad Gita.  
It uses **RAG (Retrieval-Augmented Generation)** with **LangChain**, **ChromaDB**, and **OpenAI GPT-3.5** to simulate **Krishna-like wisdom** based on user inputs.

---

## ⚙️ Tech Stack

- 🧠 **OpenAI GPT-3.5** – Language generation  
- 🛠️ **LangChain** – Chaining prompts and tools  
- 📚 **ChromaDB** – Vector storage and similarity search  
- 🌐 **FastAPI** – Backend RESTful API  
- 🗂️ **Bhagavad Gita Dataset** – Verse-wise structured data  
- 🐍 **Python** – Core programming

---

## 🚀 Features

- Accepts **user’s queries or problems**
- Retrieves **relevant shlokas** using **vector search**
- Responds with:
  - Original Sanskrit Shloka
  - Hindi/English Translation
  - Explanation as per context
- Clean FastAPI-based REST API (can integrate with frontend/chat UI)

---

## 📦 Installation & Usage

```bash
# Clone the repo
git clone https://github.com/09hritik/Bhagavd-gita-chatbot.git
cd Bhagavd-gita-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn rag1:app --reload
