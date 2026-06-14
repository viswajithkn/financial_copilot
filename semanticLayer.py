import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from datetime import datetime
from dateutil.relativedelta import relativedelta
import httpx
unsafe_http_client = httpx.Client(verify=False)

current_date = datetime.now().strftime("%Y-%m-%d")
now = datetime.now()
current_month_name = now.strftime("%B")
current_year = now.year
last_month = now - relativedelta(months=1)
last_month_name = last_month.strftime("%B")
last_month_year = last_month.year

prompt =  ChatPromptTemplate.from_template("""You are a metadata extractor for a personal finance application.
Given a user question, extract filterable metadata as JSON.

Today is {current_date}. 
Current month is {current_month_name}, year {current_year}.
Last month is {last_month_name}, year {last_month_year}.

Extract ANY of these fields if present irrespective of all the fields being present:
- post_month: month name e.g. "May" (resolve "last month", "this month")
- post_year: integer e.g. 2026
- merchant_category: use exact values from taxonomy
- merchant_name: specific merchant if mentioned
- account_name: "creditCard" or "checking"
- category: "charge", "withdrawal", "deposit", "credit", "transfer"
                                           
IMPORTANT: Always extract post_month and post_year if ANY month reference 
exists in the question, regardless of whether other filters are present.
Month extraction is independent of category, merchant, or account extraction. 
When a user asks about a concept corresponding to a merchant_category,
extract that merchant_category even if the question is not explicitly
asking for spending or transactions.

VISITED MERCHANT RULE:
If the user asks:
- merchants visited
- places visited
- stores visited
- restaurants visited
You MUST filter out merchant_category ('Payslip','Tax Refund') and merchant_name ('ATM', 'Zelle')

- IMPORTANT: When a user asks about 'merchants they 'visited,' Do not return merchants representing:
- payroll deposits
- transfers
- ATM withdrawals
- bank activity
- tax refunds


Valid merchant_category values: Retail, Education, Groceries, Coffee Shop, 
Restaurant, Subscription, Ecommerce, Fitness, Clothing, Gas, Parking, 
Travel, Travel/Lodging, Software/SaaS, Sport, Mobile, Legal, Electricity, 
Rental, Payslip, Monthly Expenses, Automobile, Internet, Financial, 
Personal Care, Pharmacy, Personal, Tax Refund, Gardening
                                           
When mapping user terms to merchant_category, use these synonyms:
- "rent", "rental payment", "lease" → Rental
- "eating out", "food", "dining", "restaurants" → Restaurant
- "coffee", "cafe" → Coffee Shop
- "gym", "workout", "exercise" → Fitness
- "clothes", "clothing store" → Clothing
- "medicine", "drug store", "prescriptions" → Pharmacy
- "car", "vehicle", "auto" → Automobile
- "phone", "cell" → Mobile
- "power", "electric bill" → Electricity
- "software", "tools", "apps" → Software/SaaS
- "subscriptions", "streaming" → Subscription
- "gas station", "fuel" → Gas
- "online shopping" → Ecommerce
- "school", "tuition", "university", "course" → Education
- "doctor", "hospital", "medical" → Healthcare   

When mapping user terms to account names:
- "credit card", "credit card account", "card account", "my card" → creditCard
- "checking", "checking account", "bank account" → checking                                                                                                                              

Return ONLY valid JSON. No explanation. No markdown.
Return {{}} if no metadata found.

Examples:
"how much did I spend on dining last month" → {{"post_month": "May", "post_year": 2026, "merchant_category": "Restaurant"}}
"show me my credit card charges in March 2025" → {{"post_month": "March", "post_year": 2025, "account_name": "creditCard", "category": "charge"}}
"what did I spend at Netflix" → {{"merchant_name": "Netflix"}}
"how much did I spend this year" → {{"post_year": 2026}}
"what is my balance in May?" → {{"post_month": "May"}}

Question: {question}""").partial(
        current_date=current_date,
        current_month_name=current_month_name,
        current_year = current_year,
        last_month_year = last_month_year,
        last_month_name=last_month_name)

valid_months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
valid_accounts = ['creditCard', 'checking']

@traceable(name="extract_metadata",run_type="chain")
def extract_metadata_filter(question: str) -> dict:
    # LLM call
    llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
             http_client=unsafe_http_client
        )
    chain = (
                {
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | JsonOutputParser()
    )    

    try:    
        response = chain.invoke(question)
    except Exception as e:
        response = {}
    # Parse JSON response
    if 'post_month' in response and response['post_month'] not in valid_months:
        response.pop('post_month')
    if 'account_name' in response and response['account_name'] not in valid_accounts:
        response.pop('account_name')    

    # Return dict or {} if parsing fails
    return response