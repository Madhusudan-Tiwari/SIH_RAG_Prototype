from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list=None, conversation_context=None):
    """
    Sends question + retrieved context + previous conversation to all LLMs.
    Automatically formats prompts correctly for each provider.
    """
    context_list = context_list or []
    context_text = "\n\n".join([str(c) for c in context_list if c])

    # Flatten conversation context into a single string for Gemini/Perplexity
    conversation_text = ""
    if isinstance(conversation_context, list):
        for msg in conversation_context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_text += f"{role.upper()}: {content}\n"

    # Build the prompt
    system_prompt = (
        "You are a helpful assistant. "
        "Use the provided context (if any) to answer the question. "
        "If the answer is not in the context, say you don't know."
    )
    prompt = f"{system_prompt}\n\n"
    if context_text:
        prompt += f"Context:\n{context_text}\n\n"
    if conversation_text:
        prompt += f"Previous Conversation:\n{conversation_text}\n"
    prompt += f"User Question: {question}"

    try:
        # MultiLLM will now format this correctly for OpenAI, Gemini, Perplexity
        return multi_llm.query(prompt)
    except Exception as e:
        return f"[Error querying LLMs: {str(e)}]"
