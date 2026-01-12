import os
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

BUCKET_NAME = os.getenv("BUCKET_NAME")  # REQUIRED
DOCS_PREFIX_DEFAULT = os.getenv("DOCS_PREFIX", "docs/")  # optional prefix where PDFs already exist

# Where to store per-user FAISS in S3 (base prefix)
FAISS_USERS_PREFIX = os.getenv("FAISS_USERS_PREFIX", "faiss-users")  # e.g. faiss-users/<user_id>/...


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
    resp = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    content_list = resp["output"]["message"]["content"]
    return "\n".join([c.get("text", "") for c in content_list if "text" in c]).strip()


# ============================================================
# Per-user namespace helpers
# ============================================================
def ensure_user_namespace():
    """
    Creates a stable ID for the visitor session.
    IMPORTANT: Streamlit sessions are per browser tab.
    If they refresh, Streamlit often keeps session state,
    but if they open in new browser or incognito, they get a new ID.
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    # You can show this in UI for debugging / support
    return st.session_state.user_id


def user_prefix(user_id: str) -> str:
    return f"{FAISS_USERS_PREFIX}/{user_id}"


def s3_key(user_id: str, name: str) -> str:
    # name = "kb.faiss" / "kb.pkl" / "manifest.json"
    return f"{user_prefix(user_id)}/{name}"


def download_s3_bytes(bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def upload_s3_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def exists_s3(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ============================================================
# Chunking / Parsing
# ============================================================
def split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def load_pdf_to_docs(pdf_bytes: bytes, source_name: str) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    for d in pages:
        d.metadata = d.metadata or {}
        d.metadata["source"] = source_name

    return pages


# ============================================================
# Per-user FAISS persistence
# ============================================================
def faiss_exists_for_user(user_id: str) -> bool:
    return (
        exists_s3(BUCKET_NAME, s3_key(user_id, "kb.faiss"))
        and exists_s3(BUCKET_NAME, s3_key(user_id, "kb.pkl"))
    )


def load_faiss_for_user(user_id: str) -> Optional[FAISS]:
    if not faiss_exists_for_user(user_id):
        return None

    folder = f"/tmp/faiss_{user_id}"
    os.makedirs(folder, exist_ok=True)

    faiss_path = os.path.join(folder, "kb.faiss")
    pkl_path = os.path.join(folder, "kb.pkl")

    with open(faiss_path, "wb") as f:
        f.write(download_s3_bytes(BUCKET_NAME, s3_key(user_id, "kb.faiss")))
    with open(pkl_path, "wb") as f:
        f.write(download_s3_bytes(BUCKET_NAME, s3_key(user_id, "kb.pkl")))

    return FAISS.load_local(
        folder_path=folder,
        index_name="kb",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def save_faiss_for_user(user_id: str, vs: FAISS):
    folder = f"/tmp/faiss_{user_id}"
    os.makedirs(folder, exist_ok=True)

    vs.save_local(folder_path=folder, index_name="kb")

    faiss_path = os.path.join(folder, "kb.faiss")
    pkl_path = os.path.join(folder, "kb.pkl")

    with open(faiss_path, "rb") as f:
        upload_s3_bytes(BUCKET_NAME, s3_key(user_id, "kb.faiss"), f.read())
    with open(pkl_path, "rb") as f:
        upload_s3_bytes(BUCKET_NAME, s3_key(user_id, "kb.pkl"), f.read())


def load_manifest_for_user(user_id: str) -> Dict:
    key = s3_key(user_id, "manifest.json")
    try:
        data = download_s3_bytes(BUCKET_NAME, key)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {"indexed_sources": []}


def save_manifest_for_user(user_id: str, manifest: Dict):
    upload_s3_bytes(
        BUCKET_NAME,
        s3_key(user_id, "manifest.json"),
        json.dumps(manifest, indent=2).encode("utf-8"),
        content_type="application/json",
    )


# ============================================================
# Ingestion (Uploads + S3 docs folder) — per user
# ============================================================
def upsert_documents(vs: Optional[FAISS], chunks: List[Document]) -> FAISS:
    if vs is None:
        return FAISS.from_documents(chunks, embeddings)
    vs.add_documents(chunks)
    return vs


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


def ingest_uploaded_pdfs(user_id: str, uploaded_files, chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
    vs = load_faiss_for_user(user_id)
    total_chunks = 0

    for f in uploaded_files:
        pdf_bytes = f.getvalue()
        source_name = f"upload://{f.name}"

        pages = load_pdf_to_docs(pdf_bytes, source_name=source_name)
        chunks = split_docs(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)

        vs = upsert_documents(vs, chunks)

    save_faiss_for_user(user_id, vs)
    return len(uploaded_files), total_chunks


def ingest_s3_prefix(user_id: str, prefix: str, chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
    manifest = load_manifest_for_user(user_id)
    indexed = set(manifest.get("indexed_sources", []))

    pdf_keys = list_s3_pdfs(BUCKET_NAME, prefix)
    to_index = [k for k in pdf_keys if k not in indexed]

    if not to_index:
        return 0, 0

    vs = load_faiss_for_user(user_id)
    total_chunks = 0

    for key in to_index:
        pdf_bytes = download_s3_bytes(BUCKET_NAME, key)
        pages = load_pdf_to_docs(pdf_bytes, source_name=f"s3://{BUCKET_NAME}/{key}")
        chunks = split_docs(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)

        vs = upsert_documents(vs, chunks)
        manifest["indexed_sources"].append(key)

    save_faiss_for_user(user_id, vs)
    save_manifest_for_user(user_id, manifest)

    return len(to_index), total_chunks


# ============================================================
# RAG Answering
# ============================================================
def rag_answer(vs: FAISS, question: str, k: int = 4, max_tokens: int = 500, temperature: float = 0.2):
    docs = vs.similarity_search(question, k=k)

    context = "\n\n---\n\n".join(
        [
            f"[Source: {d.metadata.get('source','n/a')} | page: {d.metadata.get('page','n/a')}]\n{d.page_content}"
            for d in docs
        ]
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
    st.title("Private RAG (Per-User): S3 + FAISS + Bedrock")

    if not BUCKET_NAME:
        st.error("BUCKET_NAME env var is missing. Set BUCKET_NAME before running.")
        st.stop()

    user_id = ensure_user_namespace()

    st.caption(
        f"AWS_REGION={AWS_REGION} | CHAT_MODEL_ID={CHAT_MODEL_ID} | "
        f"User namespace: {user_prefix(user_id)}"
    )

    st.session_state.setdefault("kb_loaded", False)
    st.session_state.setdefault("chat_history", [])

    tabs = st.tabs(["1) Build / Update KB", "2) Chat (RAG)", "3) Reset / Delete My KB"])

    # -------------------------
    # Tab 1
    # -------------------------
    with tabs[0]:
        st.subheader("Build / Update Knowledge Base (Private to YOU)")

        colA, colB = st.columns(2)
        with colA:
            chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
        with colB:
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=800, value=200, step=50)

        st.markdown("### A) Upload multiple PDFs (incremental add)")
        uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Ingest uploaded PDFs into MY KB", disabled=(not uploaded_files)):
            with st.spinner("Chunking + embedding + updating YOUR FAISS in S3..."):
                n_files, n_chunks = ingest_uploaded_pdfs(user_id, uploaded_files, chunk_size, chunk_overlap)
            st.success(f"Indexed {n_files} file(s), added {n_chunks} chunk(s) ✅")
            st.session_state.kb_loaded = False

        st.divider()

        st.markdown("### B) Ingest PDFs already in S3 (incremental add)")
        prefix = st.text_input("S3 docs prefix", value=DOCS_PREFIX_DEFAULT)

        if st.button("Scan S3 prefix + ingest new PDFs into MY KB"):
            with st.spinner("Scanning S3 + embedding new PDFs into your private KB..."):
                n_files, n_chunks = ingest_s3_prefix(user_id, prefix, chunk_size, chunk_overlap)
            if n_files == 0:
                st.info("No new PDFs found for you (already indexed).")
            else:
                st.success(f"Indexed {n_files} new S3 PDF(s), added {n_chunks} chunk(s) ✅")
            st.session_state.kb_loaded = False

        st.divider()

        st.markdown("### My KB Status")
        manifest = load_manifest_for_user(user_id)
        st.write(f"FAISS exists for me: **{faiss_exists_for_user(user_id)}**")
        st.write(f"My manifest indexed sources: **{len(manifest.get('indexed_sources', []))}**")
        with st.expander("View my manifest (debug)"):
            st.json(manifest)

    # -------------------------
    # Tab 2
    # -------------------------
    with tabs[1]:
        st.subheader("Chat with MY Knowledge Base (Private RAG)")

        if not st.session_state.kb_loaded:
            with st.spinner("Loading your FAISS index from S3..."):
                vs = load_faiss_for_user(user_id)
            st.session_state["vs"] = vs
            st.session_state.kb_loaded = True

        vs = st.session_state.get("vs")
        if vs is None:
            st.warning("You don’t have a KB yet. Go to Tab 1 and ingest documents first.")
            st.stop()

        col1, col2, col3 = st.columns(3)
        with col1:
            k = st.number_input("Top-K chunks", min_value=1, max_value=10, value=4)
        with col2:
            max_tokens = st.number_input("Max tokens", min_value=64, max_value=2000, value=500, step=50)
        with col3:
            temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        with st.form("chat_form", clear_on_submit=True):
            q = st.text_input("Ask a question", placeholder="e.g., Summarize what I uploaded")
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

    # -------------------------
    # Tab 3: Reset/Delete
    # -------------------------
    with tabs[2]:
        st.subheader("Reset / Delete My Knowledge Base")

        st.warning(
            "This deletes *your* FAISS index + manifest in S3 for your current session namespace. "
            "This does NOT affect other users."
        )

        if st.button("Delete my KB from S3"):
            try:
                # delete kb.faiss, kb.pkl, manifest.json (ignore missing)
                for name in ["kb.faiss", "kb.pkl", "manifest.json"]:
                    key = s3_key(user_id, name)
                    try:
                        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
                    except Exception:
                        pass

                st.session_state.kb_loaded = False
                st.session_state.chat_history = []
                st.success("Deleted your KB ✅ Refresh the page to start fresh.")
            except Exception as e:
                st.error("Failed to delete KB.")
                st.code(str(e))


if __name__ == "__main__":
    main()
