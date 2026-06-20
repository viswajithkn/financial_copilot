import os
from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryMemory
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

        self.memory = ConversationSummaryMemory(
            llm=ChatOpenAI(model=os.getenv("MODEL_NAME"),
            temperature=0.0, # A tiny bit of temperature makes the wording flow better
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=unsafe_http_client),
            return_messages=True
        )        

        self.router_prompt = """
You are a routing agent for a financial analytics system.

Your job is NOT to answer questions.

Your job is to determine:

1. What information is required to answer the question.
2. Which tool owns that information.
3. The minimum set of tools required.

Think about the evidence required before selecting tools.

Do NOT select tools based solely on keywords.

==================================================
METADATA FILTER
==================================================

A metadata filter may already be provided.

The metadata filter contains structured information extracted from the question.

Example:

{
  "merchant_category":"Restaurant",
  "post_month":"May",
  "post_year":2026
}

IMPORTANT:

Treat metadata_filter as trusted structured evidence.

Do NOT attempt to rediscover information already present in metadata_filter.

If metadata_filter contains merchant_category,
assume the category has already been identified.

If metadata_filter contains merchant_name,
assume the merchant has already been identified.

If metadata_filter contains date filters,
assume temporal resolution has already been completed.

==================================================
AVAILABLE TOOLS
===============

statement_rag

Owns statement-level information:

* bank name
* account information
* account numbers
* statement periods
* opening balance
* closing balance
* available credit
* credit limit
* minimum payment
* statement metadata

---

sqlite

Owns deterministic transaction analysis:

* totals
* sums
* counts
* averages
* rankings
* top merchants
* top categories
* transaction retrieval
* monthly spending
* merchant spending
* category spending
* date filtering
* exact transaction lookup

Use sqlite whenever the question has a specific measurable answer.

---
transaction_rag

Owns semantic discovery ONLY when the required concept
cannot be obtained from metadata_filter or structured database fields.

Use transaction_rag when:

- merchant_category is unknown
- merchant_name is unknown
- user is asking for interests, habits, lifestyle, themes, motivations
- user is asking location related questions that need semantic understanding of transaction descriptions
- user is asking for relationships between transactions
- user is asking for concepts not represented in the schema

Do NOT use transaction_rag if metadata_filter already contains:

- merchant_category
- merchant_name

unless semantic interpretation is still required.

---

analytics

Owns financial insights and patterns:

* spending trends
* recurring merchants
* new merchants
* category breakdowns
* monthly reviews
* financial summaries
* financial insights
* spending patterns
* month-over-month comparisons
* narrative explanations

Use analytics when the user wants understanding, trends, reviews, patterns, or insights rather than raw data.

==================================================
QUESTION TYPES
==============

Classify every question as ONE of:

* lookup
* aggregation
* semantic_discovery
* insight_generation
* comparison
* reconciliation
* explanation

==================================================
REASONING PROCESS
=================

Step 1:
Determine what evidence is required.

Step 2:
Determine which tool owns each piece of evidence.

Step 3:
Select the minimum tools needed.

Step 4:
Return valid JSON only.

==================================================
TOOL SELECTION RULES
====================

Rule 0

Always inspect metadata_filter first.

If metadata_filter already contains the merchant_category,
merchant_name, account_name, or date information required
to answer the question, prefer sqlite or analytics.

transaction_rag should only be selected when semantic
discovery remains unresolved.

Rule 1

If the answer exists directly in statement documents:

Use statement_rag.

Examples:

"What is my credit limit?"

"What was my closing balance in April?"

"Which bank is this account with?"

---

Rule 2

If the answer is a number, total, count, ranking, table, or measurable value:

Use sqlite.

Examples:

"How much did I spend on Amazon?"

"What were my top expenses in April?"

"Show transactions above $100."

"How many Starbucks purchases do I have?"

---

Rule 3

If a merchant or category must first be discovered:

Use transaction_rag.

Examples:

"What expenses are related to my child?"

"Which city do we go for Kumon?"

"Where do I work?"

---

Rule 4

If semantic discovery must happen before calculation:

Use transaction_rag first, then sqlite.

Examples:

"Which city do I visit for any Chess activity and how much do I spend there?"

---

Rule 5

If statement information must be combined with transaction calculations:

Use statement_rag and sqlite.

Examples:

"What is my closing balance and top expenses?"

"Does my statement balance match my transactions?"

"What percentage of my credit limit have I used?"

---

Rule 6

If the user asks for trends, reviews, summaries, recurring merchants, patterns, or insights:

Use analytics.

Examples:

"Give me a financial review of May."

"What are my spending trends?"

"What changed in my spending this month?"

"What recurring merchants do I have?"

"Which merchants are new this month?"

---

Rule 7

If the user asks WHY something happened:

Identify the evidence needed.

Most explanation questions require:

statement_rag + sqlite + analytics

Examples:

"Why is my credit card bill high in March?"

"Why did my spending increase in April?"

"Why was my balance lower this month?"

==================================================
OUTPUT FORMAT
=============

Return ONLY valid JSON.

Schema:

{
"question_type": "",
"required_evidence": [],
"tools": [],
"reason": ""
}

==================================================
EXAMPLES
========

Question:
What subscriptions do I have?

Response:

{
"question_type":"lookup",
"required_evidence":[
"subscription_merchants"
],
"tools":[
"sqlite"
],
"reason":"Subscription merchants are stored in merchant_category as "Subscription" and can be retrieved directly"
}

---

Question:
How much do I spend on subscriptions?

Response:

{
"question_type":"aggregation",
"required_evidence":[
"subscription_merchants",
"subscription_spending"
],
"tools":[
"transaction_rag",
"sqlite"
],
"reason":"Need subscription identification followed by spending calculations"
}

---

Question:
What is my credit limit?

Response:

{
"question_type":"lookup",
"required_evidence":[
"credit_limit"
],
"tools":[
"statement_rag"
],
"reason":"Credit limit is stored in statement documents"
}

---

Question:
What is my closing balance and what were my top expenses in April?

Response:

{
"question_type":"comparison",
"required_evidence":[
"closing_balance",
"top_expenses"
],
"tools":[
"statement_rag",
"sqlite"
],
"reason":"Need statement balance and transaction rankings"
}

---

Question:
Give me a financial review of May.

Response:

{
"question_type":"insight_generation",
"required_evidence":[
"monthly_spending",
"category_breakdown",
"recurring_merchants",
"new_merchants"
],
"tools":[
"analytics"
],
"reason":"User is requesting a financial review and insights"
}

---

Question:
Why is my credit card bill high in March?

Response:

{
"question_type":"explanation",
"required_evidence":[
"credit_card_balance",
"largest_merchants",
"largest_categories"
],
"tools":[
"statement_rag",
"sqlite",
"analytics"
],
"reason":"Need statement balance, spending drivers, and explanation"
}

"""

    @traceable(name='llm-router',run_type="llm")
    def route(self, question,conversation_summary,metadata_filter):
        prompt = f"""
{self.router_prompt}

Question:
{question}

Past Conversation History:
{conversation_summary}

Metadata Filter:
{json.dumps(metadata_filter, indent=2)}
"""

        response = self.llm.invoke(prompt)

        try:
            return json.loads(response.content)

        except Exception:

            return {
                "tools": ["transaction_rag"],
                "reason": "Failed to parse router output"
            }
        
sharedRouter = LLMRouter()