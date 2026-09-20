# Intelligent HR RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that answers HR policy 
questions using company documents, built for [competition/course name].

## 🎯 What it does
- Answers employee questions about leave policy, reimbursements, 
  code of conduct, etc. using only company HR documents
- Refuses to answer out-of-scope questions (guardrails)

## 🛠️ Tech Stack
- LangChain (orchestration)
- FAISS (vector database)
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Groq (LLM inference - openai/gpt-oss-20b)
- Python, Jupyter/Kaggle Notebooks

## 🏗️ Architecture
1. PDF documents loaded and chunked (RecursiveCharacterTextSplitter)
2. Chunks embedded and stored in FAISS vector store
3. Retriever fetches top-k relevant chunks per query
4. Guardrail classifier checks if question is in-scope
5. LLM generates answer grounded only in retrieved context

## 📊 Result
Scored 75.29 on the evaluation leaderboard (Rank #22)

## 🚀 How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Add your Groq API key to `.env`
3. Run the notebook cells in order
