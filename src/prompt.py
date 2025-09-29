def build_financial_prompt(context: str, question: str) -> str:
    """
    Construct a best-practice prompt for financial data RAG pipeline.

    Args:
        context (str): Retrieved context from the vector store.
        question (str): User's question.
    Returns:
        str: Formatted prompt for LLM input.
    """
    return (
        "You are a financial data expert. Use only the provided data to answer. "
        "If the answer is not present in the data, say 'I do not know based on the provided data.' "
        "Cite relevant data points and avoid speculation.\n"
        "Format your answer with bullet points for clarity.\n"
        "\nUse the following pieces of information to answer the user's question.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "\nAnswer the question and provide additional helpful information, "
        "based on the pieces of information, if applicable. Be succinct.\n"
        "Responses should be properly formatted to be easily read."
    )
