import os
from dotenv import load_dotenv
load_dotenv()

import httpx
unsafe_http_client = httpx.Client(verify=False)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from semanticLayer import build_chroma_filter


question_rewrite_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0.0, # A tiny bit of temperature makes the wording flow better
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=unsafe_http_client
)

prompt = ChatPromptTemplate.from_template("" \
"You are a conversation-aware query rewriter. " \
"Given: " \
"1. Previous conversation: {conversation_summary} " \
"2. Current user question: {question} " \
"Rewrite the current question into a fully self-contained question. " \
"Examples: " \
"Previous: Which places did I travel to? " \
"Answer: You traveled to Zion National Park, Buck Wild and Autocamp. " \
"Current: When did I visit Zion? " \
"Rewrite: When did I visit Zion National Park? " \
"Current: How much did I spend there? " \
"Rewrite: How much did I spend at Zion National Park? Return only the rewritten question.").partial()

reference_prompt = ChatPromptTemplate.from_template(
    "Determine whether the user's question requires resolving a previous conversational reference. "
    "Given: "
    "1. Previous conversation: {conversation_summary} "
    "2. Current user question: {question} "
    "Examples requiring resolution: "
    "- there "
    "- it "
    "- that "
    "- those "
    "- which one"
    "- them"
    "- again"
    "- last one "
    "If the question is standalone and understandable "
    "without prior context, return a JSON: {{\"requires_resolution\": false}} "
    "Otherwise return: "
    "{{\"requires_resolution\": true}}"
).partial()

@traceable(name="question_rewrite_llm",run_type="chain")

def reWriteQuestion(question,conversation_summary):
    chain = (
                {
                    "question": RunnablePassthrough(),
                    "conversation_summary": RunnablePassthrough(),
                }
                | prompt
                | question_rewrite_llm
                | StrOutputParser()
    )    

    chain_reference = (
                {
                    "question": RunnablePassthrough(),
                    "conversation_summary": RunnablePassthrough(),
                }
                | reference_prompt
                | question_rewrite_llm
                | JsonOutputParser()
    )        

    response_reference = chain_reference.invoke({'question':question,'conversation_summary':conversation_summary})
    if response_reference['requires_resolution']:
        try:    
            response = chain.invoke({'question':question,'conversation_summary':conversation_summary})
        except Exception as e:
            response = ''
    else:
        response = question

    return response