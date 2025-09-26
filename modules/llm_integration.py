from modules.llm_fallback import MultiLLM

llm_client = MultiLLM()

def query_llm_with_context(question, context_texts):
    context = "\n\n".join(context_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm_client.query(prompt)
