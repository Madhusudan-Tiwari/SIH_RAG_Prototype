from modules.llm_fallback import MultiLLM

multi_llm = MultiLLM()

def query_llm_with_context(question, context_list=None, conversation_context=None):
    context_list = context_list or []
    # Join the top K context documents into a single string, with clear delimiters
    context_text = "\n\n--- DOCUMENT CONTEXT ---\n\n" + "\n\n".join([str(c) for c in context_list if c]) + "\n\n------------------------\n"

    # Start with a strong system prompt to enforce context usage
    prompt = "You are a specialized RAG assistant. Your primary goal is to answer the user's question ONLY based on the text provided in the 'DOCUMENT CONTEXT' section below. If the answer is not found in the context, clearly state that you cannot answer based on the provided documents.\n"
    
    # Add the extracted context
    if context_list:
        prompt += context_text
    else:
        # If no documents were retrieved, inform the LLM (unlikely now)
        prompt += "\n\n[NO RELEVANT DOCUMENTS WERE RETRIEVED TO ANSWER THIS QUESTION]\n\n"

    # Add conversation history
    if conversation_context:
        prompt += "\n--- CONVERSATION HISTORY --UNNECESSARY IN PRODUCTION ---" # Simplified for production 
        # Removed full history logging to minimize prompt length unless needed for multi-turn.
    
    # Add the final question
    prompt += f"User Query: {question}\n\nAssistant Answer:"

    try:
        # Revert to calling the API
        answer = multi_llm.query(prompt)
        
        # Explicit check for common LLM API failure strings
        if answer.startswith("[Gemini failed:") or not answer.strip():
            return f"Error: LLM returned an empty or failed response. Check API key validity or prompt length."
            
        return answer
        
    except Exception as e:
        # Return the error message explicitly
        return f"[FATAL API ERROR]: Failed to get response from Gemini. Check network connection or API key. Error: {str(e)}"
