import os
from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from langsmith import traceable

import httpx
unsafe_http_client = httpx.Client(verify=False)


class LLMRouter:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
             http_client=unsafe_http_client
        )

        self.router_prompt = """
You are a routing agent for a financial analytics system.

Available tools:

statement_rag
- Bank name
- Account information
- Statement metadata
- Account numbers
- Credit cards
- Statement periods

transaction_rag
- Semantic merchant search
- Transaction detail and context
- Semantic financial analysis

sqlite
- Question has a specific measurable answer
- User wants a number not a narrative
- Totals
- Sums
- Counts
- Spending calculations
- Monthly analysis
- Exact transaction lookup
- Date analysis

analytics: 
- Question asks for overview, review, summary, or insights
- Question requires combining multiple data points
- User wants patterns, trends, or recommendations, insights, trends, top merchants, category breakdowns, 
  recurring payments, subscriptions, financial review, spending patterns

Routing rules for analytics:
- "give me insights" → analytics
- "financial review" → analytics
- "top merchants" → analytics
- "spending trends" → analytics
- "what subscriptions do I have" → analytics
- "recurring payments" → analytics
- "category breakdown" → analytics
- "how is my cash flow" → analytics
- "monthly summary" → analytics

Rules:

1. Use statement_rag for account and statement questions.
2. Use sqlite for how_much, calculations, dates, counts, totals, sum, average and transaction analysis. Merchant names are present in description column.
3. Use transaction_rag for semantic financial reasoning OR Merchant/category is unknown.
4. Use BOTH transaction_rag and sqlite when a question asks for mathematical calculations 
(totals, sums, counts) on a broad category, abstract concept (e.g., "software", "chess-related", "subscriptions", "eating out"). 
You must use transaction_rag first to identify the merchant names, then sqlite to compute the math.
5. Use BOTH transaction_rag and sqlite for Questions about trends, habits, patterns
6. Use statement_rag + sqlite together when:
- Question asks about balance AND transactions (reconciliation)
- Question asks why a bill is high (statement for total, RAG for detail)
- Question connects closing balance to specific spending
- Question asks about credit limit vs actual spending
7. Beyond the above rules use both ONLY when necessary.
8. Use analytics when:
- questions are about insights
- questions are about subscriptions
- questions are about recurring transactions
- questions are about monthly summary
- questions are about financial review
9. Can the answer be a single number or table? → sqlite
Does the answer require narrative and interpretation? → analytics
10. Return ONLY JSON.

Return ONLY valid JSON.

Schema:

{
  "tools": ["tool1", "tool2"],
  "reason": "short explanation"
}

Examples:

Question:
Which bank do I have an account with?

Response:
{"tools":["statement_rag"],
 "reason":"Bank information is stored in statement documents"}

Question:
where did I eat last month??

Response:
{"tools":["transaction_rag"],
 "reason":"Requires semantic identification of dining/restaurants"}

Question:
How much have I spent on Amazon?

Response:
{"tools":["sqlite"],
 "reason":"Requires aggregation of transactions"}

Question:
What subscriptions do I spend the most money on?

Response:
{"tools":["transaction_rag","sqlite"],
 "reason":"Need semantic identification followed by spending calculations"}

Question:
What recurring subscriptions do I have and how much have I spent on them?

Response:
{
  "tools":["sqlite"],
  "reason":"Need subscription identification and spending calculations"
}

Question:
Which chess related expenses cost me the most money?

Response:
{
  "tools":["transaction_rag","sqlite"],
  "reason":"Need semantic identification of chess expenses and aggregation"
}

Question: 
why is my credit card bill high in March?
{"tools": ["statement_rag", "sqlite"], 
   "reasoning": "needs statement total and merchants for which expenses are high"}

Question: 
Give me a subscription review?
{"tools": ["sqlite"], 
   "reasoning": "needs sql querying to identify subscriptions and call appropriate functions"}

Question:
Do you have any monthly trends?

Response:
{"tools":["analytics"],
 "reason":"Requires monthly analysis of transactions"}   
  

"""

    @traceable(name='llm-router',run_type="llm")
    def route(self, question):

        prompt = f"""
{self.router_prompt}

Question:
{question}
"""

        response = self.llm.invoke(prompt)

        try:
            return json.loads(response.content)

        except Exception:

            return {
                "tools": ["transaction_rag"],
                "reason": "Failed to parse router output"
            }