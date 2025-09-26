from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list, conversation_context=None):
    """
    Sends question + retrieved context + previous conversation to LLM
    """
    # Combine top-k context
    context_text = "\n".join(context_list)
    if conversation_context:
        context_text = conversation_context + "\n" + context_text

    # Construct final prompt
    final_prompt = f"{context_text}\nQuestion: {question}"

    # Query the multi-LLM fallback
    answer = multi_llm.query(final_prompt)
    return answer