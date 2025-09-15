import os
from typing import List

import streamlit as st

# LangChain core & community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import Ollama

# FinBERT encoder
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


# ---------------------------
# Streamlit UI configuration
# ---------------------------
st.set_page_config(page_title="Finance RAG (FinBERT + FAISS + Llama3)", layout="wide")
st.title("Finance Retrieval QA")
st.caption("Enter your links in separate columns, then ask a question. Uses FinBERT embeddings, FAISS, and Llama 3 via Ollama.")


# ---------------------------
# Sidebar: URL inputs + params
# ---------------------------
with st.sidebar:
    st.header("Controls")

    st.subheader("Model Settings")
    llama_model = st.text_input("Ollama model", value="llama3:8b")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    num_predict = st.number_input("Max tokens (num_predict)", min_value=64, max_value=2048, value=500, step=32)

    st.subheader("Index Settings")
    chunk_size = st.number_input("Chunk size", min_value=256, max_value=4000, value=1000, step=64)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=10)
    k = st.number_input("Top-k chunks", min_value=1, max_value=10, value=2, step=1)

    build_btn = st.button("Process URLs", type="primary")


# ------------------------------------
# FinBERT encoding utilities (cached)
# ------------------------------------
@st.cache_resource(show_spinner=False)
def get_finbert():
    tokenizer_local = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model_local = AutoModel.from_pretrained("ProsusAI/finbert")
    return tokenizer_local, model_local


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask


def encode_finbert(texts: List[str], batch_size: int = 8, max_length: int = 128) -> np.ndarray:
    tokenizer_local, model_local = get_finbert()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer_local(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model_local(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])  # (batch, hidden)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings).cpu().numpy()


class FinBERTEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = encode_finbert(texts)
        embs = np.asarray(embs, dtype="float32")
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        embs = encode_finbert([text])
        embs = np.asarray(embs, dtype="float32")
        return embs[0].tolist()


@st.cache_data(show_spinner=False)
def load_urls(urls: List[str]) -> List[Document]:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data


@st.cache_resource(show_spinner=False)
def build_faiss_from_docs(docs_in: List[Document], chunk_size_val: int, chunk_overlap_val: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val)
    chunks = splitter.split_documents(docs_in)
    texts = [doc.page_content for doc in chunks]
    metas = [doc.metadata for doc in chunks]
    documents = [Document(page_content=texts[i], metadata=metas[i]) for i in range(len(texts))]
    return FAISS.from_documents(documents, FinBERTEmbeddings())


def build_chain(store: FAISS, model_name: str, temp: float, max_tokens: int, k_val: int):
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k_val})
    llm = Ollama(model=model_name, temperature=temp, num_predict=int(max_tokens))
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    return chain


# ---------------------------
# Main panel
# ---------------------------
st.subheader("Article Links")
num_urls = st.number_input("Number of URLs", min_value=1, max_value=10, value=3, step=1)
cols = st.columns(int(num_urls))
for i, c in enumerate(cols):
    c.text_input(f"URL {i+1}", key=f"url_{i+1}")

question = st.text_input("Question", placeholder="e.g., summarize KR Choksey's report on Tata Motors")

if build_btn:
    with st.spinner("Loading URLs and building index..."):
        urls = [st.session_state.get(f"url_{i+1}", "").strip() for i in range(int(num_urls))]
        urls = [u for u in urls if u]
        if not urls:
            st.warning("Please provide at least one URL.")
            st.stop()

        raw_docs = load_urls(urls)
        if len(raw_docs) == 0:
            st.error("No content could be loaded from the provided URLs.")
            st.stop()

        store = build_faiss_from_docs(raw_docs, chunk_size, chunk_overlap)
        st.session_state["faiss_store_ready"] = True
        st.session_state["faiss_store_obj"] = store
        st.success("Index built successfully.")


run_btn = st.button("Get Answer", type="secondary")

if run_btn:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not st.session_state.get("faiss_store_ready"):
        st.warning("Please process URLs first.")
        st.stop()

    store: FAISS = st.session_state["faiss_store_obj"]

    with st.spinner("Generating answer..."):
        chain = build_chain(store, llama_model, temperature, num_predict, int(k))
        result = chain.invoke({"question": question})

    st.subheader("Answer")
    st.write(result.get("answer", ""))

    # Sources can be a string of new-line separated URLs; handle both list and string
    sources = result.get("sources")
    st.subheader("Sources")
    if isinstance(sources, list):
        for s in sources:
            if s:
                st.markdown(f"- [{s}]({s})")
    elif isinstance(sources, str):
        parts = [p.strip() for p in sources.split("\n") if p.strip()]
        for s in parts:
            st.markdown(f"- [{s}]({s})")
    else:
        st.write("No sources returned.")


st.markdown("---")
st.caption("Tip: If you plan to commit the FAISS index, ensure files remain under GitHub's 100 MB per-file limit, or use Git LFS.")


