import sqlite3
import glob
from datetime import datetime
from vectorDB import createVectorDB
from createStructuredData import createStructuredData
from createTransactionVectorDB import createTransactionVectorDB

statementFiles = glob.glob("C:/Courses/Rag/BankStatementRag/rawData/*.pdf")
conn = sqlite3.connect("transactions_db/bank_transactions.db")

cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_date TEXT,
    post_month TEXT,
    trans_date TEXT,
    trans_month TEXT,
    account_name TEXT,
    category TEXT,
    ref TEXT,
    description TEXT,
    amount REAL
)
""")

conn.commit()

transactions, docs = createStructuredData(statementFiles)
for t in transactions:
    date_object = datetime.strptime(f"{t['post_date']}/26", "%m/%d/%y")
    post_month_name = date_object.strftime("%B") 
    date_object = datetime.strptime(f"{t['trans_date']}/26", "%m/%d/%y")
    trans_month_name = date_object.strftime("%B")    
    cursor.execute(
        """
        INSERT INTO transactions
        (post_date, post_month,trans_date,trans_month, account_name, category, ref, description, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            t["post_date"],
            post_month_name,
            t["trans_date"],
            trans_month_name,
            t["account_name"],
            t["category"],
            t["ref"],
            t["description"],
            t["amount"],

        )
    )

conn.commit()
file_vectorStore = createVectorDB(statementFiles)
transaction_vectorStore = createTransactionVectorDB(docs)
