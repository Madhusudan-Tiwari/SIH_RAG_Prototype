import streamlit as st
import openai
import requests

# Optional: for Gemini
try:
    from google import genai
except ImportError:
    genai = None  # Will raise an error if used without install

class MultiLLM:
    def __init__(self):
        self.llms = [
            {"name": "openai", "func": self.query_openai},
            {"name": "gemini", "func": self.query_gemini},
            {"name": "perplexity", "func": self.query_perplexity}
        ]

    # ----------------- OpenAI -----------------
    def query_openai(self, prompt):
        """Query OpenAI GPT model using API key from Streamlit secrets."""
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI failed: {e}")
            raise

    # ----------------- Gemini -----------------
    def query_gemini(self, prompt):
        """Query Gemini API using API key from Streamlit secrets."""
        if genai is None:
            raise ImportError("google-genai library not installed. pip install google-genai")
        api_key = st.secrets["GEMINI_API_KEY"]
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini failed: {e}")
            raise

    # ----------------- Perplexity -----------------
    def query_perplexity(self, prompt):
        """Query Perplexity API using API key from Streamlit secrets."""
        api_key = st.secrets["PERPLEXITY_API_KEY"]
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.post(
                "https://api.perplexity.ai/v1/query",
                headers=headers,
                json={"query": prompt}
            )
            response.raise_for_status()
            return response.json().get("answer", "[No answer]")
        except Exception as e:
            print(f"Perplexity failed: {e}")
            raise

    # ----------------- Fallback -----------------
    def query(self, prompt):
        """
        Try each LLM in order; fallback to next if one fails.
        Returns the first successful answer or '[All LLMs failed]'.
        """
        for llm in self.llms:
            try:
                answer = llm["func"](prompt)
                if answer:
                    return answer
            except Exception:
                continue
        return "[All LLMs failed]"
