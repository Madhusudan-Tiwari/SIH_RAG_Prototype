from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list=None, conversation_context=None):
    """
    Sends question + retrieved context + previous conversation to LLM.
    """
    context_list = context_list or []
    context_text = "\n\n".join([str(c) for c in context_list if c])

    system_prompt = (
        "You are a helpful assistant. "
        "Use the provided context (if any) to answer the question. "
        "If the answer is not in the context, say you don't know."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Add previous conversation
    if isinstance(conversation_context, list):
        messages.extend(conversation_context)

    # Add retrieved context
    if context_text:
        messages.append({"role": "assistant", "content": f"Context:\n{context_text}"})

    # Add the user question
    messages.append({"role": "user", "content": question})

    try:
        return multi_llm.query(messages)
    except Exception as e:
        return f"[Error querying LLMs: {str(e)}]"
