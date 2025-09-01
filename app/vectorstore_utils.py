from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

def build_faiss_store(chunks: List[str]):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(chunks, embedder)

def search_similar_docs(store, query: str, top_k: int = 3):
    return store.similarity_search(query, k=top_k)
