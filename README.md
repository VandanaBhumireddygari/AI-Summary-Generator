# 📄 AI Summary Generator – Chat with PDFs using AWS Bedrock

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to **upload PDFs and ask natural-language questions** about their content.
Built using **AWS Bedrock**, **FAISS**, **Streamlit**, and **Docker**.

This project demonstrates a **real-world GenAI architecture** used in modern data and ML platforms.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f9f1ec68-fcca-4b31-bae7-e12a978d0a3e" />



## 🚀 Features

* Upload PDF documents via a web UI
* Automatic text extraction and chunking
* Semantic search using vector embeddings
* Context-aware question answering
* Uses **AWS Bedrock** for embeddings and LLM inference
* Dockerized for easy local execution
* Model-agnostic (swap models via environment variables)

---

## 🏗️ Architecture Overview


PDF Upload
   ↓
Text Extraction (PyPDFLoader)
   ↓
Text Chunking
   ↓
Embedding Generation (Amazon Titan)
   ↓
FAISS Vector Store
   ↓
User Question
   ↓
Similarity Search (Top-K Chunks)
   ↓
LLM (AWS Bedrock)
   ↓
Answer (Grounded, No Hallucinations)

Pattern used: **Retrieval-Augmented Generation (RAG)**

---

## ☁️ AWS Infrastructure Setup

This project includes real-world cloud infrastructure commonly used in production GenAI systems:

- Created an **EC2 instance** to host the Streamlit + Docker application
- Created an **S3 bucket** to store:
  - PDF documents
  - FAISS index files
  - Manifest metadata for incremental ingestion
- Created an **IAM Role** with least-privilege access:
  - `bedrock:InvokeModel`
  - `bedrock:Converse`
  - `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
- Attached the IAM role to the EC2 instance
- Application authenticates using **IAM role credentials (no hardcoded keys)**

This mirrors how GenAI applications securely interact with Bedrock and storage services in production.

---

## 📂 Project Structure

| File | Purpose |
|------|--------|
| `admin.py` | Single-PDF RAG demo (quick validation) |
| `rag.py` | Multi-PDF shared knowledge base (S3-backed) |
| `multi_user.py` | Multi-user, per-user isolated KB (production-ready) |

---

## 🧠 Models (AWS Bedrock)

**Embeddings**
- `amazon.titan-embed-text-v2:0`

**LLM (default)**
- `mistral.ministral-3-8b-instruct`

➡️ Models can be swapped using environment variables without code changes.

---

## 🐳 Run Locally or on EC2 (Docker)

### Build
```bash
docker build -t pdf-reader .
````

### Run

```bash
docker run --rm \
  -e AWS_REGION=us-west-2 \
  -e AWS_DEFAULT_REGION=us-west-2 \
  -e CHAT_MODEL_ID=mistral.ministral-3-8b-instruct \
  -v ~/.aws:/root/.aws \
  -p 8083:8083 \
  pdf-reader


Open in browser:

```
http://localhost:8083
```

---

## 🔐 AWS Requirements

* AWS account with **Bedrock enabled**
* IAM role or user with:

  * `bedrock:InvokeModel`
  * `bedrock:Converse`
  * S3 read/write permissions
* Credentials via:

  * IAM role attached to EC2 (recommended)
  * or local AWS CLI configuration

---

## 🔮 Future Enhancements

* OCR support for scanned PDFs
* Streaming LLM responses
* User authentication and authorization
* Replace FAISS with OpenSearch / Pinecone
* IaC using Terraform or CloudFormation

---

## 👩‍💻 Author

**Vandana Bhumireddygari**
Data Engineer | Cloud | GenAI | AWS | Snowflake

