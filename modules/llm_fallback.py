import streamlit as st
try:
    from google import genai
except ImportError:
    genai = None  # Will raise an error if used without install


class MultiLLM:
    def __init__(self):
        if genai is None:
            raise ImportError("google-genai package not installed. Install via `pip install google-genai`.")
        if "GEMINI_API_KEY" not in st.secrets:
            raise ValueError("GEMINI_API_KEY missing in Streamlit secrets")

        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        self.model = "gemini-2.5-flash-lite"

    def query_gemini(self, prompt: str) -> str:
        """
        Send prompt string to Gemini and return text response.
        """
        try:
            # Wrap prompt in a list of strings as Gemini expects string input
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            # response.text may be a string or list
            if isinstance(response.text, list):
                return "\n".join(response.text)
            return response.text
        except Exception as e:
            return f"[Gemini failed: {e}]"

    def query(self, prompt: str) -> str:
        return self.query_gemini(prompt)
