from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import httpx
import chromadb
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
unsafe_http_client = httpx.Client(verify=False)

def createVectorDB(statementFiles):
    embeddings = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # chunk size (characters)
        chunk_overlap=50,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []
    for file in statementFiles:
        loader = PyPDFLoader(file)
        docs = loader.load()
        filename = Path(file).stem
        parts = filename.split(" ")
        date_part = parts[0]
        if date_part[2:4] in ['28','29','30','31']:
            accountType = "checking"
        else:
            accountType = "creditCard"        

        date = datetime.strptime(date_part, "%m%d%y")
        statement_month = date.strftime("%B")
        statement_year = date.year
        bank_name = parts[1]          

        for doc in docs:
            doc.metadata["post_month"] = statement_month
            doc.metadata["post_year"] = statement_year
            doc.metadata["bank_name"] = bank_name
            doc.metadata["account_name"] = accountType
        all_docs.extend(docs)

    split_docs = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        http_client=unsafe_http_client
    )

    if not os.path.exists("./chroma_db"):
        vectorstore = Chroma.from_documents(
            documents=split_docs, 
            embedding=embeddings,
            collection_name="bank_statements",  # Specify your collection name here
            persist_directory="./chroma_db"
        )
    else:
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="bank_statements",  # Specify your collection name here
            persist_directory="./chroma_db"
        )

    return vectorstore