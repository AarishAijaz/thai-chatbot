# vector.py (FAISS version)
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os

# Load PDF
pdf_path = "ThaiRecipes.pdf"
loader = PyMuPDFLoader(pdf_path)
pages = loader.load()

# Create documents with metadata (optional)
documents = [
    Document(page_content=page.page_content, metadata={"source": f"page_{i}"})
    for i, page in enumerate(pages)
]

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Create FAISS index
vector_store = FAISS.from_documents(documents, embedding=embeddings)

# Create retriever
retreiver = vector_store.as_retriever(search_kwargs={"k": 10})
