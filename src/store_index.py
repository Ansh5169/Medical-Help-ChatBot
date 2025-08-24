from src.helper import load_pdf, text_splitter, download_huggingface_embeddings
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
pinecone_api_key=os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=pinecone_api_key

extractedData= load_pdf(data='Data/')
text_chunks=text_splitter(extractedData)
embeddings=download_huggingface_embeddings()
docsearch= PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name="medicalbot",
    embedding=embeddings,
)