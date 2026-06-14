import json

def needs_repair(tool: str, result: str) -> bool:
    failure_signals = [
        "i don't know",
        "no data found",
        "no transactions found",
        "cannot be determined",
        "no information available"
    ]
    return any(signal in result.lower() for signal in failure_signals)

def relax_filter(metadata_filter: dict, attempt: int) -> dict:
    if not metadata_filter or "$and" not in metadata_filter:
        return metadata_filter    
    relaxed = metadata_filter.copy()
    conditions = relaxed['and']
    if attempt == 1:
        # First retry: remove merchant_category, keep date
        # relaxed.pop('merchant_category', None)
        # relaxed.pop('merchant_name', None)
        conditions = [c for c in conditions if c not in ['merchant_category','merchant_name']]
    elif attempt == 2:
        # Second retry: remove all filters
        conditions = []
    
    if not conditions:
        return None

    return {
        "$and": conditions
    }

def repair_question(question: str, tool: str, attempt: int) -> str:
    # LLM call to rephrase the question
    prompt = f"""
    The following question returned no results from a financial {tool}.
    Rephrase it to be broader and more likely to find relevant data.
    Keep the same intent but use more general terms.
    Attempt {attempt + 1} of 2.
    
    Original: {question}
    Return only the rephrased question, nothing else.
    """

    return repair_question
