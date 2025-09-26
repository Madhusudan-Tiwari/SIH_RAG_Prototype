from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list=None, conversation_context=None):
    """
    Sends question + retrieved context + previous conversation to LLM.
    Handles empty contexts and conversation gracefully.
    """
    # Ensure context_list is always a list
    context_list = context_list or []
    context_text = "\n".join([str(c) for c in context_list if c])  # skip None/empty

    # Add previous conversation if available
    if conversation_context:
        context_text = f"{conversation_context}\n{context_text}" if context_text else conversation_context

    # Construct final prompt
    final_prompt = f"{context_text}\nQuestion: {question}" if context_text else f"Question: {question}"

    try:
        # Query the multi-LLM fallback
        answer = multi_llm.query(final_prompt)
    except Exception as e:
        # Return a fallback message if all LLMs fail
        answer = f"[Error querying LLMs: {str(e)}]"

    return answer
