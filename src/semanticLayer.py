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

EARLIEST_DATA_MONTH = "November"
EARLIEST_DATA_YEAR = 2025

prompt =  ChatPromptTemplate.from_template("""You are a metadata extractor for a personal finance application.
Given a user question, extract filterable metadata as JSON.
                                           
Use the Past Conversation History to fill in missing information if the current question is a follow-up.

---
PAST CONVERSATION HISTORY:
{conversation_summary}                                           

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
                                           
IMPORTANT: Always extract post_month and post_year if ANY month reference which is not a date range,
exists in the question, regardless of whether other filters are present.
Month extraction is independent of category, merchant, or account extraction. 
When a user asks about a concept corresponding to a merchant_category,
extract that merchant_category even if the question is not explicitly
asking for spending or transactions.

DATE RANGE RULE:
If the question implies a range rather than a single month, extract 
post_month_start/post_year_start and post_month_end/post_year_end 
instead of post_month/post_year.

Range trigger phrases:
- "up to [month]" / "through [month]" / "until [month]" 
  → start = earliest available data, end = mentioned month
- "since [month]" / "from [month] onwards" 
  → start = mentioned month, end = current month
- "from [month] to [month]" / "between [month] and [month]" 
  → start = first month, end = second month
- "last N months" 
  → start = N months before current, end = current month
- "average each month", "average monthly", "monthly average" 
  → ALWAYS implies a range, even without explicit start date
  → if no start mentioned, default start = earliest available data 
    (use {earliest_data_month} {earliest_data_year})
- "this year", "in 2026", "year to date" 
  → start = January of that year, end = current month or December

If a range is detected, do NOT include post_month/post_year — 
use post_month_start/post_year_start and post_month_end/post_year_end instead.                                               

Examples:
"how much do I spend on average each month for gas up to May 2026" 
→ {{"merchant_category": "Gas", "post_month_start": "November", "post_year_start": 2025, "post_month_end": "May", "post_year_end": 2026}}

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

When a user mentions a specific merchant:

Examples:
- Kumon
- Amazon
- Netflix
- Starbucks

You MUST add merchant_name.                                    

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

UMBRELLA CATEGORY RULE:
Some user terms refer to multiple merchant_category values at once.
When this happens, return merchant_category as a LIST of valid categories,
not a single string.

Umbrella mappings:
- "utilities", "utility bills" → ["Electricity", "Internet", "Mobile"]
- "bills", "monthly bills" → ["Electricity", "Internet", "Mobile", "Rental", "Subscription"]
- "shopping" → ["Retail", "Ecommerce", "Clothing"]
- "transportation", "transport" → ["Gas", "Automobile", "Travel"]
- "health", "healthcare" → ["Pharmacy", "Personal Care"]

If the user's term maps to exactly one category, return a single string as before.
If it maps to an umbrella of multiple categories, return a list.

Examples:
"what utilities do I have and who are the vendors" 
→ {{"merchant_category": ["Electricity", "Internet", "Mobile"]}}

"how much did I spend on bills last month"
→ {{"post_month": "May", "post_year": 2026, "merchant_category": ["Electricity", "Internet", "Mobile", "Rental", "Subscription"]}}                                           

Income-related mappings:

- "income" → category = deposit
- "salary" → merchant_category = Payslip
- "paycheck" → merchant_category = Payslip
- "payroll" → merchant_category = Payslip
- "wages" → merchant_category = Payslip
- "earnings" → category = deposit
- "compensation" → merchant_category = Payslip
- "bonus" → category = deposit
- "refund" → merchant_category = Tax Refund
- "tax refund" → merchant_category = Tax Refund
- "interest income" → merchant_category = Financial
- "investment income" → merchant_category = Financial
- "income sources" → category = deposit                                             

When mapping user terms to account names:
- "credit card", "credit card account", "card account", "my card" → creditCard
- "checking", "checking account", "bank account" → checking
- "income", "salary" - Payslip
- "tax returns", "tax refund" - Tax Refund                                                                                                                       

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
        last_month_name=last_month_name,
        earliest_data_month=EARLIEST_DATA_MONTH,
        earliest_data_year=EARLIEST_DATA_YEAR
        )

valid_months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
valid_accounts = ['creditCard', 'checking']

def get_month_year_range(start_month: str, start_year: int, end_month: str, end_year: int) -> list:
    start_date = datetime(start_year, valid_months.index(start_month) + 1, 1)
    end_date = datetime(end_year, valid_months.index(end_month) + 1, 1)
    
    pairs = []
    current = start_date
    while current <= end_date:
        pairs.append({"post_month": current.strftime("%B"), "post_year": current.year})
        current += relativedelta(months=1)
    
    return pairs

def build_chroma_filter(metadata_filter: dict) -> dict:
    if not metadata_filter:
        return None
    
    chroma_filter = {}

    if 'post_month_start' in metadata_filter and 'post_month_end' in metadata_filter:
        pairs = get_month_year_range(
            metadata_filter['post_month_start'],
            metadata_filter['post_year_start'],
            metadata_filter['post_month_end'],
            metadata_filter['post_year_end']
        )
        chroma_filter['$or'] = [
            {"$and": [{"post_month": p["post_month"]}, {"post_year": p["post_year"]}]}
            for p in pairs
        ]    
    # Single month
    if 'post_month' in metadata_filter:
        chroma_filter['post_month'] = metadata_filter['post_month']
    
    if 'post_year' in metadata_filter:
        chroma_filter['post_year'] = metadata_filter['post_year']
    
    if 'merchant_category' in metadata_filter:
        val = metadata_filter['merchant_category']
        if isinstance(val, list):
            chroma_filter['merchant_category'] = {"$in": val}
        else:
            chroma_filter['merchant_category'] = val        
    
    if 'account_name' in metadata_filter:
        chroma_filter['account_name'] = metadata_filter['account_name']
    
    if 'merchant_name' in metadata_filter:
        chroma_filter['merchant_name'] = metadata_filter['merchant_name']
        
    if not chroma_filter:
        return None
        
    # If there is only 1 condition, return it directly
    if len(chroma_filter) == 1:
        return chroma_filter    
    
    return {
        "$and": [
            {key: value} for key, value in chroma_filter.items()
        ]
    }

@traceable(name="extract_metadata",run_type="chain")
def extract_metadata_filter(question: str,conversation_summary) -> dict:
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
                    "conversation_summary": RunnablePassthrough(),
                }
                | prompt
                | llm
                | JsonOutputParser()
    )    

    try:    
        response = chain.invoke({'question':question,'conversation_summary':conversation_summary})
    except Exception as e:
        response = {}
    # Parse JSON response
    if 'post_month' in response and response['post_month'] not in valid_months:
        response.pop('post_month')
    if 'account_name' in response and response['account_name'] not in valid_accounts:
        response.pop('account_name')    

    # Return dict or {} if parsing fails
    return response