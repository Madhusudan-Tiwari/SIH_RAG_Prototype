import streamlit as st
import openai
from openai import OpenAI as OAI_Client # Import the modern client to handle Perplexity errors cleanly
from perplexity import Perplexity
from perplexity import APIStatusError, AuthenticationError as PPLX_AuthError # Perplexity specific errors
from google import genai
from google.genai.errors import APIError as GeminiAPIError


# --- Configuration Constants ---
PPLX_MODEL = "sonar"
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"

class MultiLLM:
    def __init__(self):
        self.llms = [
            # Prioritize models based on speed/cost/preference
            {"name": "Gemini", "func": self.query_gemini}, 
            {"name": "OpenAI", "func": self.query_openai},
            {"name": "Perplexity", "func": self.query_perplexity}
        ]

    # ----------------- 1. OpenAI (Legacy client for compatibility with your original code) -----------------
    def query_openai(self, prompt):
        """Query OpenAI GPT model using the legacy API."""
        openai.api_key = st.secrets.get("OPENAI_API_KEY")
        if not openai.api_key:
             raise ValueError("OPENAI_API_KEY not found in Streamlit secrets.")
             
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except openai.error.AuthenticationError as e:
            raise RuntimeError(f"OpenAI failed (Auth Error): Check API key/billing.")
        except Exception as e:
            raise RuntimeError(f"OpenAI failed: {e}")

    # ----------------- 2. Gemini (Modern google-genai client) -----------------
    def query_gemini(self, prompt):
        """Query Gemini API using the modern genai client."""
        if genai is None:
            raise ImportError("google-genai library not installed.")
            
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY not found in Streamlit secrets.")

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            # Check for error in response object (common for invalid key/quota in modern client)
            if not response.text:
                 raise ValueError("Gemini returned an empty response. Check model access/quota.")
                 
            return response.text.strip()
        except GeminiAPIError as e:
            # Catches authentication, resource, and general API errors
            raise RuntimeError(f"Gemini failed (API Error): {e}")
        except Exception as e:
            raise RuntimeError(f"Gemini failed: {e}")

    # ----------------- 3. Perplexity (Official perplexity SDK) -----------------
    def query_perplexity(self, prompt):
        """Query Perplexity API using the official SDK."""
        api_key = st.secrets.get("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in Streamlit secrets.")
        
        try:
            # Initialize the official Perplexity client
            client = Perplexity(api_key=api_key)
            
            response = client.chat.completions.create(
                model=PPLX_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except PPLX_AuthError as e:
            raise RuntimeError(f"Perplexity failed (Auth Error): Check API key/credits.")
            
        except APIStatusError as e:
            # Catches 400 (Bad Request/Invalid Model), 429 (Rate Limit), etc.
            raise RuntimeError(f"Perplexity failed (Status {e.status_code}): {e.message}")
            
        except Exception as e:
            raise RuntimeError(f"Perplexity failed: {e}")

    # ----------------- Fallback Logic -----------------
def query(self, prompt):
    if not prompt or prompt.strip() == "":
        return "[Please enter a question.]"

    failed_llms = []
    for llm in self.llms:
        try:
            st.info(f"Attempting to query {llm['name']}...")
            with st.spinner(f"Waiting for {llm['name']}..."):
                answer = llm["func"](prompt)

            # --- Normalize all responses to string ---
            if isinstance(answer, list):
                answer = " ".join(str(x) for x in answer if x)

            if answer and str(answer).strip():
                st.success(f"✅ Response received from {llm['name']}.")
                return str(answer).strip()

        except Exception as e:
            failed_llms.append(f"{llm['name']}: {str(e)}")
            continue

    st.error("❌ All LLMs failed to provide a response.")
    return f"[All LLMs failed. Detailed Errors:\n- {';\n- '.join(failed_llms)}]"


# --- Streamlit App Entry Point ---

def main():
    st.set_page_config(page_title="Multi-LLM Fallback Demo", layout="wide")
    st.title("🌐 Multi-LLM Query with Fallback")
    st.caption("Queries in order: Gemini, OpenAI, Perplexity. Falls back on failure.")

    # Show a warning if Streamlit secrets are not configured (common issue)
    if not st.secrets.get("OPENAI_API_KEY") or \
       not st.secrets.get("GEMINI_API_KEY") or \
       not st.secrets.get("PERPLEXITY_API_KEY"):
       st.warning("⚠️ Please ensure your API keys (OPENAI_API_KEY, GEMINI_API_KEY, PERPLEXITY_API_KEY) are correctly configured in your Streamlit secrets file (`.streamlit/secrets.toml`).")


    # --- User Interface ---
    prompt = st.text_area("Ask your question:", "What are the three most popular LLM providers right now, and why?", height=150)
    
    # Initialize the LLM class only once
    llm_runner = MultiLLM()

    if st.button("Run Query", type="primary"):
        if prompt:
            st.subheader("🤖 AI Response")
            
            # Get the response from the query logic
            response = llm_runner.query(prompt)
            
            # Display the final output
            st.markdown(response)
        else:
            st.error("Please enter a question to run the query.")

if __name__ == "__main__":
    main()