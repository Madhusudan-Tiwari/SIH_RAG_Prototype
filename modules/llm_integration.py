from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list=None, conversation_context=None):
    context_list = context_list or []
    context_text = "\n\n".join([str(c) for c in context_list if c])

    prompt = "You are a helpful assistant.\n"
    if context_text:
        prompt += f"Here is some context from documents:\n{context_text}\n\n"
    if conversation_context:
        for msg in conversation_context:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt += f"User: {question}"

    try:
        return multi_llm.query(prompt)  # now this sends ONE string prompt to Gemini
    except Exception as e:
        return f"[Error querying LLMs: {str(e)}]"
