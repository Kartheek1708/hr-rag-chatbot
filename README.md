# 💼 HR Assistant Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers employee HR
questions using only the company's own policy documents. Built for the
Gen AI Project 2: Intelligent RAG challenge — scored 75.29 on the
leaderboard.

🚀 **Live demo:** https://hr-rag-chatbot-4g4j7dr38tyokxmcuglsqh.streamlit.app/

## 🎯 What it does

- Answers questions about leave policy, reimbursements, code of conduct,
  benefits, etc., using only the provided HR PDFs (no hallucinated answers)
- Includes a guardrail classifier that politely declines out-of-scope
  questions (e.g. general knowledge, unrelated topics)

## 🛠️ Tech stack

- **LangChain** – orchestration
- **FAISS** – vector store for semantic search
- **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
- **Groq** (`openai/gpt-oss-20b`) – LLM for answer generation
- **Streamlit** – web UI

## 🏗️ Architecture

1. HR policy PDFs are loaded and split into chunks (`RecursiveCharacterTextSplitter`)
2. Chunks are embedded and stored in a FAISS vector index
3. On each question, the retriever fetches the top-k most relevant chunks
4. A guardrail prompt classifies the question as IN_SCOPE / OUT_OF_SCOPE
5. If in scope, the LLM generates an answer grounded only in the retrieved context

## 🚀 Run it locally

```bash
git clone <this-repo-url>
cd hr-rag-app
pip install -r requirements.txt
```

Add your Groq API key as an environment variable:

```bash
export GROQ_API_KEY=your_key_here      # macOS/Linux
setx GROQ_API_KEY "your_key_here"      # Windows
```

Put your HR policy PDFs in a folder named `hr_docs/` in the project root,
then run:

```bash
streamlit run app.py
```

## ☁️ Deploy for free (Streamlit Community Cloud)

1. Push this repo to GitHub (make sure `hr_docs/` PDFs are included, and
   your API key is **not** hardcoded anywhere)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **New app**, select this repo and `app.py`
4. Under **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy — you'll get a public link like `yourapp.streamlit.app`

## 📁 Project structure

```
hr-rag-app/
├── app.py              # Streamlit app (RAG pipeline + UI)
├── requirements.txt
├── hr_docs/            # HR policy PDFs (add your own)
└── README.md
```
