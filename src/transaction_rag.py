import os
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from dateutil.relativedelta import relativedelta
import httpx
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langsmith import traceable
from semanticLayer import build_chroma_filter
from statement_rag import format_context

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
persist_directory="./chroma_db_transactions"
)

current_date = datetime.now().strftime("%Y-%m-%d")
now = datetime.now()
current_month_name = now.strftime("%B")
current_year = now.year
last_month = now - relativedelta(months=1)
last_month_name = last_month.strftime("%B")
last_month_year = last_month.year

chroma_data = vectorstore.get()
all_documents = [
    Document(page_content=text, metadata=meta) 
    for text, meta in zip(chroma_data['documents'], chroma_data['metadatas'])
]
# 3. Create the BM25 retriever using the extracted documents
bm25_retriever = BM25Retriever.from_documents(all_documents)
bm25_retriever.k = 3

TRANSACTION_RAG_PROMPT = ChatPromptTemplate.from_template("""
    You are a financial document analyst with context from the previous tool{otherToolContext}.
                                                          
    Use the above results to:
    - Avoid repeating information already found
    - Fill in gaps not covered by previous tools
    - Corroborate or add detail to existing findings                                                          

    Your task is to answer questions using only the provided bank statement context. 
    When answering concept-based questions (e.g. "dining", "groceries"), 
    use merchant_category from the context. For specific merchant questions, 
    use merchant_name. Only reason about merchant names if merchant_category
     is absent to map to merchants.
                                                        
    Today's date is {current_date}.
    Current month is {current_month_name} - {current_year}.
    Last month is {last_month_name} - {last_month_year}.                                                      
                                                        
    Valid merchant_category rules: Retail, Education, Groceries, Coffee Shop, Restaurant, 
    Subscription, Ecommerce, Fitness, Clothing, Gas, Parking, Travel, Travel/Lodging, Software/SaaS, 
    Sport, Mobile, Legal, Electricity, Rental, Payslip, Monthly Expenses, Automobile, 
    Internet, Financial, Personal Care, Pharmacy, Personal, Tax Refund, Gardening

    Spending is stored as negative amounts. Income as positive. 

    Never perform financial calculations.

    Never estimate totals from retrieved documents.

    You may:

    - identify merchants
    - identify categories
    - identify subscriptions
    - identify locations mentioned in merchant names
    - identify locations mentioned in transaction descriptions
    - identify recurring destinations
    - identify activities associated with merchants
    - make simple evidence-based inferences directly supported by the retrieved context

    Examples:

    Merchant:
    KUMON OF CUPERTINO LEARNING CENTER

    Valid Answer:
    You appear to go to Cupertino for Kumon.

    Merchant:
    BERKELEY CHESS SCHOOL

    Valid Answer:
    You appear to go to Berkeley for chess activities.

    Merchant:
    AUTOCAMP ZION

    Valid Answer:
    You appear to travel to Zion.

    These are considered evidence-based extractions, not inventions.

    All calculations must be delegated to SQLite.                                                                                                                  

    LOCATION EXTRACTION RULE

    Information may be embedded inside merchant names
    or transaction descriptions.

    Examples:

    "KUMON OF CUPERTINO LEARNING CENTER"
    → Cupertino

    "BERKELEY CHESS SCHOOL"
    → Berkeley

    "AUTOCAMP ZION"
    → Zion

    Extracting a location from a merchant name or description
    is considered retrieval, not invention.
                                                          
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
    """).partial(
        current_date=current_date,
        current_month_name=current_month_name,
        current_year = current_year,
        last_month_year = last_month_year,
        last_month_name=last_month_name
    )

@traceable(name="transaction_rag",run_type="chain")
def run_transaction_rag(question,metadata_filter,otherToolContext):
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
                | TRANSACTION_RAG_PROMPT
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

#     while True:
#         user_question = input("I am a finance analyst. Do you have any questions: ")
#         if user_question.lower() == "exit":
#             break
#         try:
#             answer = run_transaction_rag(user_question)
#             print("Answer:")
#             print(answer)
#         except Exception as e:
#             print(f"An error occurred: {e}")
