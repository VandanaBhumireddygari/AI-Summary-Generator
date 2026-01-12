# 📄 AI Summary Generator – Chat with PDFs using AWS Bedrock

A **production-style Retrieval-Augmented Generation (RAG)** application that allows users to upload PDFs and ask natural-language questions over their content.

Built using **AWS Bedrock, FAISS, Streamlit, and Docker**, this project demonstrates how modern GenAI systems are designed with **grounded responses, persistence, and multi-user isolation**.

---

## 🚀 Key Features

- Upload and query PDF documents via web UI
- Semantic search using vector embeddings
- Context-grounded answers (no hallucinations)
- Persistent FAISS indexes stored in S3
- Incremental ingestion using manifests
- **Per-user isolated knowledge bases**
- Fully Dockerized for local or EC2 deployment
- Model-agnostic (swap Bedrock models via environment variables)

---

## 🏗️ Architecture (RAG Pattern)

PDFs → Text Chunking → Embeddings → FAISS
↑
User Question
↓
Top-K Retrieval
↓
AWS Bedrock (LLM)
↓
Grounded Answer

Pattern used: **Retrieval-Augmented Generation (RAG)**

---

## 📂 Project Structure

| File | Purpose |
|------|--------|
| `admin.py` | Single-PDF RAG demo (fast validation) |
| `rag.py` | Multi-PDF shared knowledge base (S3-backed) |
| `multi_user.py` | Per-user isolated KB (production-ready) |

---

## 🧠 Models (AWS Bedrock)

**Embeddings**
- `amazon.titan-embed-text-v2:0`

**LLM (default)**
- `mistral.ministral-3-8b-instruct`

➡️ Models can be swapped via environment variables without code changes.

---

## 🐳 Run Locally (Docker)

### Build
```bash
docker build -t pdf-reader .
Run
docker run --rm \
  -e AWS_REGION=us-west-2 \
  -e AWS_DEFAULT_REGION=us-west-2 \
  -e CHAT_MODEL_ID=mistral.ministral-3-8b-instruct \
  -v ~/.aws:/root/.aws \
  -p 8083:8083 \
  pdf-reader


Open in browser:

http://localhost:8083

🔐 AWS Requirements

AWS account with Bedrock enabled

IAM permissions:

bedrock:InvokeModel

bedrock:Converse

AWS credentials configured locally:

aws configure

🔮 Future Enhancements

OCR support for scanned PDFs

Streaming LLM responses

Authentication and access control

Replace FAISS with OpenSearch / Pinecone
