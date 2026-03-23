from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # [Legacy] 기존 방식: UnstructuredFileLoader
    # 최신 방식: TextLoader 사용
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_EMBEDDING_MODEL"),
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    s
