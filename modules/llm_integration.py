from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list=None, conversation_context=None):
    context_list = context_list or []
    context_text = "\n\n".join([str(c) for c in context_list if c])

    system_prompt = (
        "You are a helpful assistant. Use the provided context (if any) "
        "to answer the question. If the answer is not in the context, say you don't know.\n\n"
    )

    final_prompt = system_prompt
    if conversation_context:
        final_prompt += f"Conversation history:\n{conversation_context}\n\n"
    if context_text:
        final_prompt += f"Context:\n{context_text}\n\n"
    final_prompt += f"Question: {question}"

    return multi_llm.query(final_prompt)
