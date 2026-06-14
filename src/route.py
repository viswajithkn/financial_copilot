import os
from dotenv import load_dotenv
load_dotenv()

from llmRouter import LLMRouter
from statement_rag import run_statement_rag
from transaction_rag import run_transaction_rag
from semanticLayer import extract_metadata_filter
from sqlite_agent import run_sqlite
from langsmith import traceable
from privacy import PIIMasker
from langchain_openai import ChatOpenAI
from analytics import run_analytics_agent
import httpx
unsafe_http_client = httpx.Client(verify=False)


router = LLMRouter()

VALID_TOOLS = {
    "statement_rag",
    "transaction_rag",
    "sqlite",
    "analytics"
}

synthesizer_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0.2, # A tiny bit of temperature makes the wording flow better
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=unsafe_http_client
)

MAX_RETRIES = 2

def route_question(question):
    masker = PIIMasker()

    routing = router.route(question)

    metadata_filter = extract_metadata_filter(question)

    tools = routing.get("tools", [])
    reason = routing.get("reason", "No reason provided.")
    print(f"   👉 Chosen Tools: {tools}")
    print(f"   👉 Reason: {reason}")        

    tools = [
        tool
        for tool in tools
        if tool in VALID_TOOLS
    ]

    context = {}
    for tool in tools:
        if tool == "statement_rag":
            attempt = 0
            tempResult = run_statement_rag(question,metadata_filter,context)
            context["statement_rag"] = tempResult
            raw_result = context["statement_rag"]
        elif tool == "transaction_rag":
            context["transaction_rag"] = run_transaction_rag(question,metadata_filter,context)
            raw_result = context["transaction_rag"]
        elif tool == 'sqlite':
            sql_question = build_sql_context(
                question,
                context
            )

            result = run_sqlite(sql_question)

            context["sqlite"] = result
            raw_result = context["sqlite"]
        elif tool == "analytics":
            context["analytics"] = run_analytics_agent(question, metadata_filter)            

    masked_context = {}
    for tool_name, data in context.items():
        # Convert dictionary rows/tuples into strings and hide data numbers
        masked_context[tool_name] = masker.mask_text(str(data))

    masked_final_answer = synthesize_final_answer(question, masked_context)

    # 6. RESTORE LAYER: Put the real numbers back into the final string text for the user
    results = masker.unmask_text(masked_final_answer)     

    return results

@traceable(name='llm-router',run_type="llm")
def synthesize_final_answer(question, tool_results):
    """
    Takes the original question and the dictionary of raw tool results,
    and blends them into a single, comprehensive user answer.
    """
    # If no tools were triggered or all came back empty
    if not tool_results:
        return "I couldn't gather any information to answer that question."

    # Format the raw dictionary data into a clean text block for the LLM
    context_data = ""
    for tool_name, data in tool_results.items():
        context_data += f"\n--- Data from {tool_name} ---\n{str(data)}\n"

    synthesis_prompt = f"""
You are a helpful financial assistant summarizing data for a user.

Original Question: {question}

Here is the raw data gathered from our analytics system:
{context_data}

Instructions:
1. Combine the data from all tools into a single, seamless, natural paragraph.
2. Do not say things like "According to the SQLite tool" or "Tool 1 found". The user should not know tools exist.
3. Be direct, factual, and strictly use the data provided above.
4. If the tools found conflicting data or no data, gently explain the summary clearly.

NUMERIC SAFETY RULES (HIGHEST PRIORITY)

1. Never perform arithmetic.
2. Never add, subtract, average, count, or aggregate values.
3. Never derive a total from individual transactions.
4. Never derive a total from monthly breakdowns.
5. Only report numeric values exactly as they appear in tool outputs.
6. If a total is required but not explicitly provided by a tool, say:
   "A total value was not provided by the analytics system."

Aggregation questions: "how much", "total", "sum", "average", "how many times", "count"
Semantic questions: "what", "which", "show me", "list", "tell me about", "where did I"

Conflict Resolution Rules:
1. If one tool finds specific transaction records with exact dates/amounts (e.g., PG&E charges on 12/05) 
and another tool says "No data found" or "I don't know,":
    a - prioritize the output from SQLITE tool for aggregation related questions
    b - b - prioritize transaction RAG for descriptive, narrative, or exploratory questions
2. Do not tell the user there is a "discrepancy" or that "the database says you do not pay." 
3. If both tools return conflicting non-empty results:
   - Use SQL agent result for any numeric/amount answers
   - Use transaction RAG result for descriptive/narrative answers
   - Never average or blend two different numbers


Final Answer:"""

    response = synthesizer_llm.invoke(synthesis_prompt)
    return response.content

def build_sql_context(question, context):

    prompt = f"""
Original Question:
{question}

Previous Tool Outputs:

{context}

Use the information above when generating SQL.
"""

    return prompt
