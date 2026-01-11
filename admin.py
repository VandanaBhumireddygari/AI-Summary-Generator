"""import boto3
import streamlit as st
import os
import uuid
import tempfile

from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
## import FAISS
from langchain_community.vectorstores import FAISS

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)


#s3_client
s3_client=boto3.client("s3")
BUCKET_NAME=os.getenv("BUCKET_NAME")

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True



def main():
    st.write("this is admin site for chat with pdf demo")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        request_id = str(uuid.uuid4())
        st.write(f"Request Id: {request_id}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            saved_file_name = tmp.name

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.write(f"Total Pages: {len(pages)}")

        # Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")

        if len(splitted_docs) == 0:
            st.warning("No chunks created. PDF might be scanned (no extractable text).")
            return

        # Show chunk text (not the Document object)
        st.write("===================")
        st.text(splitted_docs[0].page_content[:800])

        if len(splitted_docs) > 1:
            st.write("===================")
            st.text(splitted_docs[1].page_content[:800])
        
        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully")
        else:
            st.write("Error!! Please check logs.")





if __name__=="__main__":
    main()
"""
import os
import uuid
import tempfile

import boto3
import streamlit as st

from botocore.exceptions import ClientError

from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# ----------------------------
# AWS / Bedrock setup
# ----------------------------
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "mistral.ministral-3-8b-instruct")

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_runtime,
)


def bedrock_converse_text(model_id: str, user_text: str, max_tokens: int = 400, temperature: float = 0.2) -> str:
    """
    Calls Bedrock Converse API using messages/text schema (NOT 'prompt').
    Works with models that support converse().
    """
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {"role": "user", "content": [{"text": user_text}]}
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

    content_list = response["output"]["message"]["content"]
    return "\n".join([c["text"] for c in content_list if "text" in c]).strip()


# ----------------------------
# Helpers
# ----------------------------
def split_text(pages, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def build_vectorstore_from_pdf(uploaded_file):
    """Returns (vectorstore, debug_info_dict)."""
    request_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    total_chars = sum(len(p.page_content or "") for p in pages)
    docs = split_text(pages, chunk_size=1000, chunk_overlap=200)

    if len(docs) == 0 or total_chars == 0:
        return None, {
            "request_id": request_id,
            "pages": len(pages),
            "chunks": len(docs),
            "total_chars": total_chars,
            "error": "No extractable text found. If the PDF is scanned/image-only, OCR is required.",
        }

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore, {
        "request_id": request_id,
        "pages": len(pages),
        "chunks": len(docs),
        "total_chars": total_chars,
        "error": None,
    }


def answer_question(vectorstore: FAISS, question: str, k: int = 4):
    """Retrieve relevant chunks and ask the model to answer using ONLY those chunks."""
    docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs)]
    )

    user_text = f"""
You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""".strip()

    answer = bedrock_converse_text(
        model_id=CHAT_MODEL_ID,
        user_text=user_text,
        max_tokens=500,
        temperature=0.2,
    )

    return answer, docs


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.title("Admin: Upload PDF → Chat with it")
    st.caption(f"Region: {AWS_REGION} | Chat model: {CHAT_MODEL_ID}")

    # init session state
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("debug", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("ready", False)

    # Step 1
    st.subheader("1) Upload & Process PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    col1, col2 = st.columns([1, 1])
    with col1:
        process_clicked = st.button("Process PDF", disabled=(uploaded_file is None))
    with col2:
        reset_clicked = st.button("Reset")

    if reset_clicked:
        st.session_state.vectorstore = None
        st.session_state.debug = None
        st.session_state.chat_history = []
        st.session_state.ready = False
        st.rerun()

    if process_clicked:
        if uploaded_file is None:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Processing PDF (load → split → embed → FAISS)..."):
                vectorstore, debug = build_vectorstore_from_pdf(uploaded_file)

            st.session_state.vectorstore = vectorstore
            st.session_state.debug = debug
            st.session_state.ready = (vectorstore is not None and not debug.get("error"))

            if debug.get("error"):
                st.error(debug["error"])
            else:
                st.success("PDF processed successfully ✅")
                st.rerun()

    # Debug info
    if st.session_state.debug:
        dbg = st.session_state.debug
        st.caption(
            f"Request: {dbg['request_id']} | pages={dbg['pages']} | chunks={dbg['chunks']} | chars={dbg['total_chars']}"
        )

    # Step 2
    st.subheader("2) Ask questions")

    if not st.session_state.ready or st.session_state.vectorstore is None:
        st.info("Process the PDF first (Step 1). After success, Q&A will appear here.")
        return

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about your PDF",
            placeholder="e.g., Summarize it in 5 bullet points",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        try:
            with st.spinner("Searching & answering..."):
                answer, sources = answer_question(st.session_state.vectorstore, question, k=4)

            st.session_state.chat_history.append((question, answer))

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Sources (top matching chunks)"):
                for i, d in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {i}** — page: {d.metadata.get('page', 'n/a')}")
                    st.text(d.page_content[:1200])
                    st.write("---")

        except ClientError as e:
            st.error("Bedrock invocation failed.")
            st.code(str(e), language="text")
            st.info(
                "If you see throughput / inference profile errors, we will switch to a different model "
                "or use an inference profile ARN."
            )

    # Chat history
    if st.session_state.chat_history:
        st.markdown("### Chat history")
        for q, a in reversed(st.session_state.chat_history[-10:]):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.write("---")


if __name__ == "__main__":
    main()


