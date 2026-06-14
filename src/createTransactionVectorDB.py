from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings 
import httpx
import os

from dotenv import load_dotenv

load_dotenv()
unsafe_http_client = httpx.Client(verify=False)

def createTransactionVectorDB(docs):
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        http_client=unsafe_http_client
    )

    if not os.path.exists("./chroma_db_transactions"):
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings,
            collection_name="bank_statements",  # Specify your collection name here
            persist_directory="./chroma_db_transactions"
        )
    else:
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="bank_statements",  # Specify your collection name here
            persist_directory="./chroma_db_transactions"
        )

    return vectorstore
