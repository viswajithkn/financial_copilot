# Personal Financial Analytics Assistant

## Overview

Personal Financial Analytics Assistant is a production-inspired AI system that combines deterministic analytics, retrieval-augmented generation (RAG), semantic search, and agentic orchestration to answer complex questions about personal finances.

The project was built to explore how modern AI systems can combine structured data, unstructured documents, and intelligent routing while maintaining accuracy, explainability, privacy, and observability.

Unlike traditional RAG systems that rely primarily on vector retrieval, this platform uses structured analytics as the primary source of truth and augments it with retrieval and semantic reasoning when appropriate.

---

## Key Features

### Deterministic Financial Analytics

A SQLite-based analytics engine serves as the primary source of truth for:

* Spending analysis
* Merchant-level aggregation
* Category-level aggregation
* Monthly and yearly summaries
* Trend analysis
* Transaction lookup
* Statistical calculations

Financial calculations are performed directly in SQL rather than by the LLM to improve reliability and reduce hallucinations.

---

### Statement RAG

Retrieval-Augmented Generation over bank statement PDFs.

Capabilities include:

* Bank identification
* Account information retrieval
* Statement period extraction
* Balance retrieval
* Credit limit retrieval
* Statement metadata lookup

---

### Transaction RAG

Semantic retrieval layer over enriched transaction records.

Capabilities include:

* Subscription discovery
* Recurring merchant detection
* Merchant categorization
* Spending pattern discovery
* Semantic financial reasoning

Examples:

* "What subscriptions do I have?"
* "What interests can you infer from my purchases?"
* "What expenses are related to chess?"

---

### Multi-Tool Orchestration

An LLM Router dynamically selects the appropriate tools based on user intent.

Supported execution patterns include:

| Query Type                | Tool Selection           |
| ------------------------- | ------------------------ |
| Account information       | Statement RAG            |
| Spending calculations     | SQLite                   |
| Subscription discovery    | Transaction RAG          |
| Spending by subscriptions | Transaction RAG + SQLite |
| Financial reviews         | Analytics                |
| Balance reconciliation    | Statement RAG + SQLite   |

---

### Semantic Metadata Layer

Transaction records are enriched with structured metadata.

Additional attributes include:

* Merchant Name
* Merchant Category
* Account Type
* Transaction Type
* Spending Taxonomy

This enables users to ask questions naturally without needing exact transaction descriptions.

Examples:

* "How much did I spend on education?"
* "What software subscriptions do I pay for?"
* "How much do I spend on my child's activities?"

---

### Privacy Layer

Sensitive financial information is masked before being sent to the LLM.

Examples:

* Account numbers
* Card numbers
* Personally identifiable information

This provides an additional security layer while maintaining answer quality.

---

### Observability

Integrated with LangSmith for:

* Trace visualization
* Agent execution tracking
* Retrieval inspection
* Prompt debugging
* Tool execution monitoring

---

## System Architecture

User Query

↓

LLM Router

↓

┌─────────────────────────────────┐
│ Tool Selection │
└─────────────────────────────────┘

↓

Statement RAG SQLite Analytics Transaction RAG

↓

Tool Outputs

↓

Response Synthesizer

↓

Final Answer

---

## Example Queries

### Statement Questions

* Which bank do I have an account with?
* What is my credit limit?
* What was my closing balance in May?

### Analytics Questions

* How much did I spend on groceries last month?
* What were my top expenses in April?
* Show all transactions above $100 in March.

### Semantic Questions

* What subscriptions do I have?
* What interests can you infer from my spending?
* What recurring payments do I make?

### Multi-Tool Questions

* How much do I spend on subscriptions?
* Why is my credit card bill high in March?
* What is my closing balance and what were my top expenses in April?

---

## Technology Stack

### LLMs

* OpenAI GPT-4o-mini

### Orchestration

* LangChain
* LangSmith

### Retrieval

* Chroma Vector Database
* OpenAI Embeddings

### Data Storage

* SQLite

### Language

* Python

---

## Key Engineering Learnings

### Deterministic Systems First

Financial calculations should be performed by structured systems rather than LLM reasoning whenever possible.

### Data Enrichment Matters

Raw transaction descriptions provide limited analytical value. Enriching transactions with merchant metadata and semantic categories significantly improves user experience.

### Retrieval Is Not Enough

Many financial questions require a combination of retrieval, structured analytics, and reasoning. Effective orchestration is often more important than the retrieval system itself.

### Privacy Must Be Designed In

Sensitive financial data should be protected before interacting with LLMs.

---

## Future Enhancements

* Verification Layer for cross-source validation
* Session Memory
* Financial Insight Discovery Agent
* Automated Monthly Financial Reviews
* Evaluation Framework
* Cost and Latency Monitoring
* Query Planning Agent
* Automated Query Repair Loop

---

## Project Goal

The objective of this project is to explore how production-grade AI systems can combine structured analytics, semantic retrieval, orchestration, privacy controls, and observability to deliver accurate and explainable financial insights.

