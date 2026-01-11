> **Use case:** Chat with PDFs using AWS Bedrock + FAISS + Streamlit

---

# 📄 AI Summary Generator – Chat with PDFs using AWS Bedrock

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to **upload PDFs and ask natural-language questions** about their content.
Built using **AWS Bedrock**, **FAISS**, **Streamlit**, and **Docker**.

This project demonstrates a **real-world GenAI architecture** used in modern data and ML platforms.


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

```
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
```

This follows a **Retrieval-Augmented Generation (RAG)** pattern.

---

## 🧠 Vector Store: FAISS

**FAISS (Facebook AI Similarity Search)** is used as the vector database for fast semantic search.

### Why FAISS?

* Extremely fast similarity search
* Lightweight and CPU-optimized
* No external service dependency
* Ideal for local development and prototypes

### How FAISS is used

1. PDF text is split into overlapping chunks
2. Each chunk is converted into a vector using embeddings
3. Vectors are stored in a FAISS index (in memory)
4. User queries are embedded
5. FAISS performs Top-K similarity search
6. Relevant chunks are passed to the LLM

**Index type:** Flat (exact search)
**Storage:** In-memory (per session)

> 🔮 Future enhancement: Persist FAISS index to S3 or replace with OpenSearch / Pinecone for production scale.

---

## 🤖 Model Definitions (AWS Bedrock)

This project uses **AWS Bedrock** for both embeddings and text generation.

---

### 🔹 Embedding Model

**Model:**

```
amazon.titan-embed-text-v2:0
```

**Purpose:**

* Converts text into dense vector representations
* Optimized for semantic similarity tasks

**Why Titan Embeddings?**

* High-quality embeddings
* Low latency
* Cost-effective
* Native AWS Bedrock integration

Used for:

* Embedding PDF chunks
* Embedding user questions
* Powering FAISS similarity search

---

### 🔹 Chat / Generation Model

**Default Model:**

```
mistral.ministral-3-8b-instruct
```

**Invocation:**

* Uses Bedrock `converse` API
* Message-based schema (role + content)

**Why this model?**

* Instruction-tuned for Q&A
* Faster and cheaper than very large models
* Good reasoning for document summarization

**Supported alternatives (swap via env var):**

* `anthropic.claude-3.5-sonnet`
* `anthropic.claude-3-haiku`
* `amazon.nova-lite`
* `amazon.nova-premier`

> 🔁 Model switching requires **no code changes**, only an environment variable update.

---

## 🔁 Retrieval-Augmented Generation (RAG)

This system enforces **grounded answers**:

* Only retrieved PDF chunks are provided to the model
* If the answer is not in the document, the model says:

  > *“I don’t know based on the provided context.”*

This significantly reduces hallucinations and improves trustworthiness.

---

## 🖥️ User Interface

* Built using **Streamlit**
* Two-step workflow:

  1. Upload & process PDF
  2. Ask questions interactively
* Displays:

  * Extracted chunk counts
  * Answer
  * Source text snippets used for answering

---

## 🐳 Docker Setup

### Build the image

```bash
docker build --no-cache -t pdf-reader-admin .
```

### Run the app

```bash
docker run --rm \
  -e AWS_REGION=us-west-2 \
  -e AWS_DEFAULT_REGION=us-west-2 \
  -e CHAT_MODEL_ID=mistral.ministral-3-8b-instruct \
  -v ~/.aws:/root/.aws \
  -p 8083:8083 \
  pdf-reader-admin
```

### Open in browser

```
http://localhost:8083
```

---

## 🔐 AWS Requirements

* AWS account with **Bedrock enabled**
* IAM user/role with:

  * `bedrock:InvokeModel`
  * `bedrock:Converse`
* AWS credentials configured locally:

```bash
aws configure
```

---

## 📦 Tech Stack

| Component        | Technology            |
| ---------------- | --------------------- |
| UI               | Streamlit             |
| LLM              | AWS Bedrock           |
| Embeddings       | Amazon Titan          |
| Vector Store     | FAISS                 |
| PDF Parsing      | LangChain PyPDFLoader |
| Language         | Python 3.11           |
| Containerization | Docker                |

---

## 🧪 Example Questions

* “Summarize this document”
* “What are the key skills listed?”
* “What is the main objective of the resume?”
* “Which technologies are mentioned?”

---

## 🔮 Future Enhancements

* Persist FAISS index to S3
* Add OCR support for scanned PDFs
* Multi-PDF support
* User authentication
* Replace FAISS with OpenSearch / Pinecone
* Streaming responses

---

## 👩‍💻 Author

**Vandana Bhumireddygari**
Data Engineer | Cloud | GenAI | AWS | Snowflake

📎 GitHub: [https://github.com/VandanaBhumireddygari](https://github.com/VandanaBhumireddygari)

---
