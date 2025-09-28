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

    # Build up conversation as text
    conversation_text = ""
    if isinstance(conversation_context, list):
        for msg in conversation_context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_text += f"\n[{role.upper()}]: {content}"

    # Final assembled prompt
    full_prompt = f"{system_prompt}\n\n"
    if context_text:
        full_prompt += f"Context:\n{context_text}\n\n"
    if conversation_text:
        full_prompt += f"Conversation so far:\n{conversation_text}\n\n"
    full_prompt += f"User: {question}"

    try:
        answer = multi_llm.query(full_prompt)   # ✅ now a string
    except Exception as e:
        answer = f"[Error querying LLMs: {str(e)}]"

    return answer
