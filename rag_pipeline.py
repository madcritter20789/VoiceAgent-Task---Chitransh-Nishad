from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import os

def load_docs(path):
    loader = PyPDFLoader(path)
    return loader.load()

# rag_pipeline.py

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def embed_docs(docs):
    # Use HuggingFace's all-MiniLM model (local + free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db


def retrieve_context(query: str, db, k: int = 3) -> str:
    return "\n".join([doc.page_content for doc in db.similarity_search(query, k=k)])
