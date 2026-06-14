import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import httpx
from langsmith import traceable
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

unsafe_http_client = httpx.Client(verify=False)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",
    http_client=unsafe_http_client
)

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

llm = init_chat_model(model=os.getenv("MODEL_NAME"), temperature=0, api_key=os.getenv("OPENAI_API_KEY"), http_client=unsafe_http_client)
vectorstore = Chroma(
embedding_function=embeddings,
collection_name="bank_statements",  # Specify your collection name here
persist_directory="./chroma_db"
)

chroma_data = vectorstore.get()
all_documents = [
    Document(page_content=text, metadata=meta) 
    for text, meta in zip(chroma_data['documents'], chroma_data['metadatas'])
]
# 3. Create the BM25 retriever using the extracted documents
bm25_retriever = BM25Retriever.from_documents(all_documents)
bm25_retriever.k = 3

STATEMENT_RAG_PROMPT = ChatPromptTemplate.from_template("""
    You are a financial document analyst with context from the previous tool{otherToolContext}.

    Use the above results to:
    - Avoid repeating information already found
    - Fill in gaps not covered by previous tools
    - Corroborate or add detail to existing findings                                                        

    Your task is to answer questions using only the provided bank statement context.
    When asked about "closing balance" or "ending balance", 
    look for terms like: "ending balance", "closing balance", 
    "final balance", "balance forward", or the last balance 
    figure shown on the statement page.
    Rules:

    1. Use only the supplied context.
    2. Do not invent information.
    3. If the answer cannot be determined from the context, say:
    "I don't know."

    4. Be concise and factual.
    5. If multiple accounts or statement periods are present,
    mention them explicitly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

def format_context(context: dict) -> str:
    if not context:
        return "No previous tool results."
    formatted = ""
    for tool, result in context.items():
        formatted += f"\n{tool.upper()} RESULT:\n{result}\n"
    return formatted

def build_chroma_filter(metadata_filter: dict) -> dict:
    if not metadata_filter:
        return None
    
    chroma_filter = {}
    
    # Single month
    if 'post_month' in metadata_filter:
        chroma_filter['post_month'] = metadata_filter['post_month']
    
    # Multiple months (last 3 months etc.)
    if 'post_months' in metadata_filter:
        chroma_filter['post_month'] = {"$in": metadata_filter['post_months']}
    
    if 'post_year' in metadata_filter:
        chroma_filter['post_year'] = metadata_filter['post_year']
    
    if 'merchant_category' in metadata_filter:
        chroma_filter['merchant_category'] = metadata_filter['merchant_category']
    
    if 'account_name' in metadata_filter:
        chroma_filter['account_name'] = metadata_filter['account_name']
    
    if 'merchant_name' in metadata_filter:
        chroma_filter['merchant_name'] = metadata_filter['merchant_name']
        
    if not metadata_filter:
        return None
        
    # If there is only 1 condition, return it directly
    if len(metadata_filter) == 1:
        return metadata_filter    
    
    return {
        "$and": [
            {key: value} for key, value in metadata_filter.items()
        ]
    }


@traceable(name="statement_rag",run_type="chain")
def run_statement_rag(question,metadata_filter,otherToolContext):
    chroma_filter = build_chroma_filter(metadata_filter)
    search_kwargs={
        "k": 5,
        "fetch_k": 10
    }
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter    
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",search_kwargs=search_kwargs
    )
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[1, 0]
    )
    chain = (
                {
                    "context": hybrid_retriever  | format_docs,
                    "question": RunnablePassthrough(),
                    "otherToolContext":lambda _: format_context(otherToolContext)
                }
                | STATEMENT_RAG_PROMPT
                | llm
                | StrOutputParser()
        )
    response = chain.invoke(question)
    return response

# if __name__ == "__main__":
#     # Ensure your OPENAI_API_KEY is set in your environment
#     if not os.getenv("OPENAI_API_KEY"):
#         print("Warning: OPENAI_API_KEY environment variable not set.")
    
#     # Define the question you want to ask your bank statement data
#     sample_question = "Which bank do I have an account with?"
    
#     print(f"Asking question: '{sample_question}'\n")
    
#     try:
#         answer = run_statement_rag(sample_question)
#         print("Answer:")
#         print(answer)
#     except Exception as e:
#         print(f"An error occurred: {e}")

