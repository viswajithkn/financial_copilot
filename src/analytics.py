import sqlite3
import os
import httpx
import pandas as pd
import json

from langsmith import traceable
from privacy import PIIMasker
from langchain_openai import ChatOpenAI

unsafe_http_client = httpx.Client(verify=False)
db_sqlite3 = sqlite3.connect("C:/Courses/Rag/BankStatementRag/transactions_db/bank_transactions.db")

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0.0, # A tiny bit of temperature makes the wording flow better
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=unsafe_http_client
)

def get_top_merchants(db,metadata_filter):
    if not metadata_filter:
        whereClause = "WHERE amount < 0 and category in ('charge','withdrawal')"
    else:
        filter_criteria = " AND ".join(
        f'{key}="{value}"'
        for key, value in metadata_filter.items()
        )
        whereClause = f"where (amount<0) AND category in ('charge','withdrawal') AND {filter_criteria}"        
    query = f"""
    SELECT
        merchant_name,
        SUM(abs(amount)) AS total_spend
    FROM transactions
    {whereClause}
    GROUP BY merchant_name
    ORDER BY total_spend DESC
    LIMIT 10
    """

    return pd.read_sql_query(query,db).to_json(orient="records", indent=4)

def get_merchant_category_breakdown(db,metadata_filter):
    if not metadata_filter:
        whereClause = "WHERE amount < 0 and category in ('charge','withdrawal')"
    else:
        filter_criteria = " AND ".join(
        f'{key}="{value}"'
        for key, value in metadata_filter.items()
        )
        whereClause = f"where (amount<0) AND category in ('charge','withdrawal') AND {filter_criteria}"        

    query = f"""
    SELECT
        merchant_category,
        SUM(abs(amount)) AS total_spend
    FROM transactions
    {whereClause}
    GROUP BY merchant_category
    ORDER BY total_spend DESC
    """

    return pd.read_sql_query(query,db).to_json(orient="records", indent=4)

def get_monthly_spending(db,metadata_filter):

    if not metadata_filter:
        whereClause = "WHERE amount < 0 and category in ('charge','withdrawal')"
    else:
        filter_criteria = " AND ".join(
        f'{key}="{value}"'
        for key, value in metadata_filter.items()
        )
        whereClause = f"where (amount<0) AND category in ('charge','withdrawal') AND {filter_criteria}"        

    query = f"""
    SELECT
        post_month,
        SUM(amount) AS total_spend
    FROM transactions
    {whereClause}
    GROUP BY post_month
    ORDER BY post_month
    """

    return pd.read_sql_query(query,db).to_json(orient="records", indent=4)

def get_new_merchants(db,metadata_filter):
    
    query = """
    SELECT
        merchant_name,
        MIN(post_month) AS first_seen
    FROM transactions where (amount<0)
    GROUP BY merchant_name
    ORDER BY first_seen DESC
    LIMIT 10
    """

    return pd.read_sql_query(query,db).to_json(orient="records", indent=4)

def get_recurring_merchants(db,metadata_filter):

    if not metadata_filter:
        whereClause = "where (amount<0)"
    else:
        filter_criteria = " AND ".join(
        f'{key}="{value}"'
        for key, value in metadata_filter.items()
        )
        whereClause = f"where (amount<0) AND {filter_criteria}"
    query = f"""
    SELECT
        merchant_name,
        COUNT(DISTINCT post_month) AS months_seen,
        SUM(amount) AS total_spend
    FROM transactions {whereClause}
    GROUP BY merchant_name
    HAVING months_seen >= 3
    ORDER BY months_seen DESC
    """

    temp_df = pd.read_sql_query(query,db)
    recurring_merchants_json = temp_df.to_json(orient="records", indent=4)
    return recurring_merchants_json


ANALYTICS_FUNCTIONS = {"get_monthly_spending": {
            "fn": get_monthly_spending,
            "description": "Returns total spending by month. Use for trend analysis, monthly comparisons, spending over time."
        },
        "get_merchant_category_breakdown": {
            "fn": get_merchant_category_breakdown,
            "description": "Returns spending broken down by merchant_category. Use for understanding where money goes."
        },
        "get_top_merchants": {
            "fn": get_top_merchants,
            "description": "Returns top merchants by total spend. Use for merchant analysis, biggest expenses."
        },
        "get_recurring_merchants": {
            "fn": get_recurring_merchants,
            "description": "Returns merchants appearing across multiple months. Use for recurring payment detection."
        }} 

@traceable(name="insight_function_selector")
def select_insight_functions(question: str) -> list:
    
    functions_desc = "\n".join([
        f"- {name}: {meta['description']}" 
        for name, meta in ANALYTICS_FUNCTIONS.items()
    ])
    
    prompt = f"""
    You are a financial insight orchestrator.
    Select which analytics functions to call to answer the user's question.
    
    Available functions:
    {functions_desc}
    
    Rules:
    - Only select functions relevant to the question
    - If identify_subscriptions is selected, always include get_recurring_merchants first
    - Order matters — list functions in execution order
    - Return minimum functions needed, do not over-fetch
    - Return ONLY a JSON array of function names, nothing else
    
    Examples:
    "Give me a monthly review" → ["get_monthly_spending", "get_merchant_category_breakdown", "get_top_merchants"]
    "Give me a full financial review" → ["get_monthly_spending", "get_merchant_category_breakdown", "get_top_merchants", "get_recurring_merchants"]
    
    Question: {question}
    """
    
    # LLM call
    response = llm.invoke(prompt)
    functions = json.loads(response.content)
    return functions

def execute_analytics_functions(functions: list, conn, metadata_filter: dict) -> dict:
    results = {}
    
    for fn_name in functions:
        if fn_name not in ANALYTICS_FUNCTIONS:
            continue
            
        try:
            fn = ANALYTICS_FUNCTIONS[fn_name]["fn"]
            results[fn_name] = fn(conn,metadata_filter)
                
        except Exception as e:
            results[fn_name] = f"Error: {str(e)}"
    
    return results

@traceable(name="synthesize_insights")
def synthesize_insights(question: str, results: dict) -> str:
    
    # Format results clearly for LLM
    formatted_results = "\n\n".join([
        f"{fn_name.upper().replace('_', ' ')}:\n{json.dumps(result, indent=2)}"
        for fn_name, result in results.items()
    ])
    
    prompt = f"""
    You are a personal financial advisor providing clear, actionable insights.
    
    User Question: {question}
    
    Analytics Data:
    {formatted_results}
    
    Rules:
    1. Answer the question directly using only the data provided
    2. Always cite specific amounts and merchants
    3. Highlight top spending areas and patterns
    4. Flag any concerns (high spending, too many subscriptions, etc.)
    5. Be concise and actionable — no fluff
    6. Never fabricate data not present above
    7. If data is missing or empty, say so explicitly
    8. Spending amounts are stored as negative — use absolute values
    
    Format:
    - Lead with direct answer to the question
    - Follow with key insights from the data
    - End with 1-2 actionable recommendations if relevant
    """
    
    response = llm.invoke(prompt)
    return response.content

@traceable(name="insight_agent")
def run_analytics_agent(question: str, metadata_filter: dict = {}) -> str:
    db_sqlite3 = sqlite3.connect("C:/Courses/Rag/BankStatementRag/transactions_db/bank_transactions.db")
    functions = select_insight_functions(question)
    results = execute_analytics_functions(functions, db_sqlite3, metadata_filter)
    response = synthesize_insights(question, results)
    return response
    


