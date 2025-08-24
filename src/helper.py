from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data):
    loader=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

def text_splitter(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts=text_splitter.split_documents(documents)
    return texts

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings