from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import re


from matplotlib import text


def parse_bank_statement_with_footer_cleanup(text):
    # CRITICAL FIX: Truncate standard statement footer boilerplate text immediately if found
    footer_patterns = [r"Totals\s+\$", r"The\s+Ending\s+Daily\s+Balance\s+does", r"Monthly\s+service\s+fee"]
    for pattern in footer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            
    lines = text.strip().split('\n')
    tx_blocks = []
    current_block = []
    
    # 1. Group multi-line text by date triggers
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if re.match(r'^\s*\d{1,2}/\d{1,2}\s', line):
            if current_block:
                tx_blocks.append(" ".join(current_block))
            current_block = [line_str]
        else:
            if current_block:
                current_block.append(line_str)
    if current_block:
        tx_blocks.append(" ".join(current_block))
        
    final_table = []
    
    # 2. Extract out dates, text descriptions, and dollar columns
    for block in tx_blocks:
        date_match = re.match(r'^(\d{1,2}/\d{1,2})\s+(.*)', block)
        if not date_match:
            continue
        date = date_match.group(1)
        remainder = date_match.group(2)
        
        amounts_str = re.findall(r'(-?\d{1,3}(?:,\d{3})*\.\d{2})', remainder)
        
        description = remainder
        for a_str in amounts_str:
            description = description.replace(a_str, "", 1)
        description = re.sub(r'\s+', ' ', description).strip()
        
        deposit = ""
        withdrawal = ""
        ending_daily_balance = ""
        tx_value = None
        
        if len(amounts_str) == 2:
            tx_value = amounts_str[0]
            ending_daily_balance = amounts_str[1]
        elif len(amounts_str) == 1:
            tx_value = amounts_str[0]

        if tx_value:
            desc_lower = description.lower()
            if "from" in desc_lower or "payroll" in desc_lower or "deposit" in desc_lower:
                deposit = tx_value
            else:
                withdrawal = tx_value

        final_table.append({
            "Date": date,
            "Description": description,
            "Deposits/Additions": deposit,
            "Withdrawals/Subtractions": withdrawal,
            "Ending Daily Balance": ending_daily_balance
        })

    return final_table

def createStructuredData(statementFiles):    
    pattern_credit = re.compile(
            r"^"
            r"(?:"
                r"(?P<card_end>\d{4})\s+(?P<trans_date>\d{2}/\d{2})\s+(?P<post_date>\d{2}/\d{2})" # Line starts with Card End
                r"|"
                r"(?P<trans_date_no_card>\d{2}/\d{2})\s+(?P<post_date_no_card>\d{2}/\d{2})"       # Line starts directly with Dates
            r")\s+"
            r"(?P<ref_num>\w+)\s+"                    # Reference Number
            r"(?:"
                # Choice A: Description contains payment terms -> CREDIT
                r"(?P<description_credit>.+?(?:PAYMENT|THANK YOU|CREDIT|REFUND).+?)\s+(?P<credit>[\d,]+\.\d{1,2})$"
                r"|"
                # Choice B: Otherwise -> CHARGE
                r"(?P<description_charge>.+?)\s+(?P<charge>[\d,]+\.\d{1,2})$"
            r")"
        )

    transactions = []
    for file in statementFiles:
        if file.split('\\')[-1][2:4] in ['28','29','30','31']:
            accountType = "checking"
            loader = PyPDFLoader(file)
        else:
            accountType = "creditCard"
            pattern = pattern_credit
            loader = PyPDFLoader(file)
        docs = loader.load()
        for doc in docs:
            text = doc.page_content
            lines = text.split("\n")

            if accountType == 'creditCard':
                clean_lines = []

                for line in lines:
                    line = line.strip()

                    if not line:
                        continue

                    if "statement" in line.lower():
                        continue

                    if "page" in line.lower():
                        continue

                    clean_lines.append(line)

                for line in clean_lines:
                    match = pattern.search(line)
                    if match:
                        data = match.groupdict()
                        desc = data["description_credit"] or data["description_charge"]

                        credit_val = float(data["credit"].replace(",", "")) if data["credit"] else None
                        charge_val = float(data["charge"].replace(",", "")) if data["charge"] else None
                        post_date = data.get('post_date') or data.get('post_date_no_card')                            
                        trans_date = data.get('trans_date') or data.get('trans_date_no_card')

                        ref = data['ref_num']
                        if data["credit"]:
                            amount = credit_val
                            category = "credit"
                        else:
                            amount = -charge_val
                            category = "charge"

                        transactions.append({
                            "post_date": post_date,
                            "trans_date": trans_date,
                            "account_name": accountType,
                            "category": category,
                            "ref": ref,
                            "description": desc,
                            "amount": amount
                        })                            
            else:   
                transaction_rows = parse_bank_statement_with_footer_cleanup(text)                

                for row in transaction_rows:
                    date = row['Date']
                    withdrawal = None
                    deposit = None                    
                    if row['Withdrawals/Subtractions']:
                        withdrawal = float(row['Withdrawals/Subtractions'].replace(",", ""))
                    if row['Deposits/Additions']:
                        deposit = float(row['Deposits/Additions'].replace(",", ""))

                    desc = row['Description']
                    if withdrawal is not None:
                        amount = -withdrawal
                        category = "withdrawal"
                    elif deposit is not None:
                        amount = deposit
                        category = "deposit"
                    else:
                        continue
                    transactions.append({
                        "post_date": date,
                        "trans_date": date,
                        "account_name": accountType,
                        "category": category,
                        "ref": '',
                        "description": desc,
                        "amount": amount
                    })

    docs = []

    for t in transactions:
        docs.append(
            Document(
                page_content = f"On {t['trans_date']}, a {t['category']} of {t['amount']} occurred at {t['description']} via {t['account_name']}.",
                metadata={
                    "post_date": t["post_date"],
                    "post_month": t["post_date"][:7],
                    "trans_date": t["trans_date"],
                    "trans_month": t["trans_date"][:7],
                    "description": t["description"],
                    "account_name": t["account_name"],
                    "category": t["category"],
                    "ref": t["ref"],
                    "amount": t["amount"],
                }
            )
        )

    return transactions,docs
