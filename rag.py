import os
import io
import json
import uuid
import tempfile
from typing import List, Tuple, Optional, Dict

import boto3
import streamlit as st
from botocore.exceptions import ClientError

from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# Config
# ============================================================
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "mistral.ministral-3-8b-instruct")

# REQUIRED: where FAISS index is persisted
BUCKET_NAME = os.getenv("BUCKET_NAME")  # e.g. "bedrock-chat-with-pdf2"
FAISS_PREFIX = os.getenv("FAISS_PREFIX", "faiss-index")  # "folder" inside bucket
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "kb")   # kb.faiss + kb.pkl
MANIFEST_KEY = os.getenv("MANIFEST_KEY", f"{FAISS_PREFIX}/manifest.json")

# Optional: where your source PDFs already exist in S3
DOCS_PREFIX = os.getenv("DOCS_PREFIX", "docs/")  # e.g. "docs/"


# ============================================================
# AWS clients
# ============================================================
s3 = boto3.client("s3", region_name=AWS_REGION)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock,
)


# ============================================================
# Bedrock Converse (messages schema)
# ============================================================
def bedrock_converse_text(
    model_id: str,
    user_text: str,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> str:
    """
    Uses Bedrock Converse API (messages schema).
    This avoids the "prompt" schema errors you saw.
    """
    resp = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    content_list = resp["output"]["message"]["content"]
    return "\n".join([c.get("text", "") for c in content_list if "text" in c]).strip()


# ============================================================
# Chunking / Parsing
# ============================================================
def split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def load_pdf_to_docs(pdf_bytes: bytes, source_name: str) -> List[Document]:
    """
    Writes bytes to temp file, uses PyPDFLoader, attaches metadata.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # attach metadata
    for d in pages:
        d.metadata = d.metadata or {}
        d.metadata["source"] = source_name

    return pages


# ============================================================
# S3 helpers (FAISS persistence + manifest)
# ============================================================
def s3_key_for_index(ext: str) -> str:
    # ext is ".faiss" or ".pkl"
    return f"{FAISS_PREFIX}/{FAISS_INDEX_NAME}{ext}"


def download_s3_bytes(bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def upload_s3_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def load_manifest() -> Dict:
    """
    Tracks which docs were already indexed to avoid re-embedding same S3 PDFs.
    """
    try:
        data = download_s3_bytes(BUCKET_NAME, MANIFEST_KEY)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {"indexed_sources": []}


def save_manifest(manifest: Dict):
    upload_s3_bytes(
        BUCKET_NAME,
        MANIFEST_KEY,
        json.dumps(manifest, indent=2).encode("utf-8"),
        content_type="application/json",
    )


def faiss_exists_in_s3() -> bool:
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key_for_index(".faiss"))
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key_for_index(".pkl"))
        return True
    except Exception:
        return False


def load_faiss_from_s3() -> Optional[FAISS]:
    """
    Downloads FAISS files to /tmp and loads them.
    """
    if not faiss_exists_in_s3():
        return None

    folder = "/tmp/faiss"
    os.makedirs(folder, exist_ok=True)

    faiss_key = s3_key_for_index(".faiss")
    pkl_key = s3_key_for_index(".pkl")

    faiss_path = os.path.join(folder, f"{FAISS_INDEX_NAME}.faiss")
    pkl_path = os.path.join(folder, f"{FAISS_INDEX_NAME}.pkl")

    with open(faiss_path, "wb") as f:
        f.write(download_s3_bytes(BUCKET_NAME, faiss_key))
    with open(pkl_path, "wb") as f:
        f.write(download_s3_bytes(BUCKET_NAME, pkl_key))

    # IMPORTANT: allow_dangerous_deserialization=True is required for FAISS pickle load
    return FAISS.load_local(
        folder_path=folder,
        index_name=FAISS_INDEX_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def save_faiss_to_s3(vs: FAISS):
    """
    Saves index locally then uploads to S3.
    """
    folder = "/tmp/faiss"
    os.makedirs(folder, exist_ok=True)

    vs.save_local(folder_path=folder, index_name=FAISS_INDEX_NAME)

    faiss_path = os.path.join(folder, f"{FAISS_INDEX_NAME}.faiss")
    pkl_path = os.path.join(folder, f"{FAISS_INDEX_NAME}.pkl")

    with open(faiss_path, "rb") as f:
        upload_s3_bytes(BUCKET_NAME, s3_key_for_index(".faiss"), f.read())
    with open(pkl_path, "rb") as f:
        upload_s3_bytes(BUCKET_NAME, s3_key_for_index(".pkl"), f.read())


# ============================================================
# Ingestion (Uploads + S3 docs folder)
# ============================================================
def upsert_documents_into_kb(
    new_docs: List[Document],
    existing_vs: Optional[FAISS],
) -> FAISS:
    """
    If vs exists, add documents; else create new FAISS index.
    """
    if existing_vs is None:
        return FAISS.from_documents(new_docs, embeddings)
    else:
        existing_vs.add_documents(new_docs)
        return existing_vs


def list_s3_pdfs(bucket: str, prefix: str) -> List[str]:
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            k = item["Key"]
            if k.lower().endswith(".pdf"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def ingest_s3_prefix_into_kb(
    prefix: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[int, int]:
    """
    Reads PDFs from s3://BUCKET/DOCS_PREFIX, skips already indexed, updates KB.
    Returns (num_files_indexed, num_chunks_added)
    """
    manifest = load_manifest()
    indexed = set(manifest.get("indexed_sources", []))

    pdf_keys = list_s3_pdfs(BUCKET_NAME, prefix)
    to_index = [k for k in pdf_keys if k not in indexed]

    if not to_index:
        return 0, 0

    vs = load_faiss_from_s3()

    total_chunks = 0
    for key in to_index:
        pdf_bytes = download_s3_bytes(BUCKET_NAME, key)
        pages = load_pdf_to_docs(pdf_bytes, source_name=f"s3://{BUCKET_NAME}/{key}")
        chunks = split_docs(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)

        vs = upsert_documents_into_kb(chunks, vs)

        manifest["indexed_sources"].append(key)

    save_faiss_to_s3(vs)
    save_manifest(manifest)

    return len(to_index), total_chunks


def ingest_uploaded_pdfs_into_kb(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[int, int]:
    """
    Takes Streamlit uploaded PDFs, updates KB in S3.
    Returns (num_files, num_chunks)
    """
    vs = load_faiss_from_s3()
    total_chunks = 0

    for f in uploaded_files:
        pdf_bytes = f.getvalue()
        source_name = f"upload://{f.name}"

        pages = load_pdf_to_docs(pdf_bytes, source_name=source_name)
        chunks = split_docs(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)

        vs = upsert_documents_into_kb(chunks, vs)

    save_faiss_to_s3(vs)
    return len(uploaded_files), total_chunks


# ============================================================
# RAG Answering
# ============================================================
def rag_answer(
    vs: FAISS,
    question: str,
    k: int = 4,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> Tuple[str, List[Document]]:
    docs = vs.similarity_search(question, k=k)

    context = "\n\n---\n\n".join(
        [f"[Source: {d.metadata.get('source','n/a')} | page: {d.metadata.get('page','n/a')}]\n{d.page_content}"
         for d in docs]
    )

    user_text = f"""
You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question: {question}
""".strip()

    answer = bedrock_converse_text(
        model_id=CHAT_MODEL_ID,
        user_text=user_text,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return answer, docs


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.title("RAG Knowledge Base: S3 + FAISS + Bedrock")

    if not BUCKET_NAME:
        st.error("BUCKET_NAME env var is missing. Set BUCKET_NAME before running.")
        st.stop()

    st.caption(
        f"AWS_REGION={AWS_REGION} | CHAT_MODEL_ID={CHAT_MODEL_ID} | "
        f"S3= s3://{BUCKET_NAME}/{FAISS_PREFIX}/"
    )

    st.session_state.setdefault("kb_loaded", False)
    st.session_state.setdefault("chat_history", [])

    tabs = st.tabs(["1) Build / Update Knowledge Base", "2) Chat (RAG)"])

    # -------------------------
    # Tab 1: Build/Update KB
    # -------------------------
    with tabs[0]:
        st.subheader("Build / Update Knowledge Base (RAG Index)")

        colA, colB = st.columns(2)

        with colA:
            chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
        with colB:
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=800, value=200, step=50)

        st.markdown("### A) Upload multiple PDFs (incremental add)")
        uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Ingest uploaded PDFs into KB", disabled=(not uploaded_files)):
            with st.spinner("Chunking + embedding + updating FAISS in S3..."):
                n_files, n_chunks = ingest_uploaded_pdfs_into_kb(uploaded_files, chunk_size, chunk_overlap)
            st.success(f"Indexed {n_files} file(s), added {n_chunks} chunk(s) to the KB ✅")
            st.session_state.kb_loaded = False  # force reload in chat tab

        st.divider()

        st.markdown("### B) Ingest PDFs already in S3 (incremental add)")
        st.write("This scans an S3 prefix and only embeds PDFs not already in the manifest.")
        prefix = st.text_input("S3 docs prefix", value=DOCS_PREFIX)

        if st.button("Scan S3 prefix + ingest new PDFs"):
            with st.spinner("Scanning S3 prefix + embedding new PDFs..."):
                n_files, n_chunks = ingest_s3_prefix_into_kb(prefix, chunk_size, chunk_overlap)
            if n_files == 0:
                st.info("No new PDFs found to index (already up to date).")
            else:
                st.success(f"Indexed {n_files} new S3 PDF(s), added {n_chunks} chunk(s) ✅")
            st.session_state.kb_loaded = False

        st.divider()

        st.markdown("### KB Status")
        manifest = load_manifest()
        st.write(f"FAISS exists in S3: **{faiss_exists_in_s3()}**")
        st.write(f"Manifest indexed sources: **{len(manifest.get('indexed_sources', []))}**")
        with st.expander("View manifest (debug)"):
            st.json(manifest)

    # -------------------------
    # Tab 2: Chat
    # -------------------------
    with tabs[1]:
        st.subheader("Chat with your Knowledge Base (RAG)")

        if not st.session_state.kb_loaded:
            with st.spinner("Loading FAISS index from S3..."):
                vs = load_faiss_from_s3()
            st.session_state["vs"] = vs
            st.session_state.kb_loaded = True

        vs = st.session_state.get("vs")

        if vs is None:
            st.warning("No KB found yet. Go to Tab 1 and ingest documents first.")
            st.stop()

        col1, col2, col3 = st.columns(3)
        with col1:
            k = st.number_input("Top-K chunks", min_value=1, max_value=10, value=4)
        with col2:
            max_tokens = st.number_input("Max tokens", min_value=64, max_value=2000, value=500, step=50)
        with col3:
            temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        with st.form("chat_form", clear_on_submit=True):
            q = st.text_input("Ask a question", placeholder="e.g., What does the document say about refunds?")
            submitted = st.form_submit_button("Ask")

        if submitted and q.strip():
            try:
                with st.spinner("Retrieving relevant chunks + generating answer..."):
                    ans, sources = rag_answer(vs, q, k=int(k), max_tokens=int(max_tokens), temperature=float(temperature))

                st.session_state.chat_history.append((q, ans))
                st.markdown("### Answer")
                st.write(ans)

                with st.expander("Sources (retrieved chunks)"):
                    for i, d in enumerate(sources, 1):
                        st.markdown(f"**{i}. {d.metadata.get('source','n/a')}** (page={d.metadata.get('page','n/a')})")
                        st.text(d.page_content[:1200])
                        st.write("---")

            except ClientError as e:
                st.error("Bedrock call failed.")
                st.code(str(e), language="text")

        if st.session_state.chat_history:
            st.markdown("### Chat history")
            for qh, ah in reversed(st.session_state.chat_history[-10:]):
                st.markdown(f"**Q:** {qh}")
                st.markdown(f"**A:** {ah}")
                st.write("---")


if __name__ == "__main__":
    main()
