import streamlit as st

try:
    from google import genai
except ImportError:
    genai = None  # Will raise an error if used without install


class MultiLLM:
    def __init__(self):
        # Check if google-genai is installed
        if genai is None:
            raise ImportError("google-genai package not installed. Install via `pip install google-genai`.")

        # Ensure the API key exists in Streamlit secrets
        if "GEMINI_API_KEY" not in st.secrets:
            raise ValueError("GEMINI_API_KEY missing in Streamlit secrets. Add it in .streamlit/secrets.toml")

        # Initialize Gemini client using the secret key
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        self.model = "gemini-2.5-flash-lite"

    def query_gemini(self, prompt: str) -> str:
        """
        Sends a prompt to the Gemini model and returns the response text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            # Handle list or string responses
            if isinstance(response.text, list):
                return "\n".join(response.text)
            return response.text

        except Exception as e:
            return f"[Gemini failed: {e}]"

    def query(self, prompt: str) -> str:
        """
        Generic query method – currently same as query_gemini().
        """
        return self.query_gemini(prompt)


# Example Streamlit usage
if __name__ == "__main__":
    st.title("Gemini Query Demo")

    user_input = st.text_area("Enter your prompt:")
    if st.button("Send"):
        try:
            llm = MultiLLM()
            output = llm.query(user_input)
            st.write("### Response:")
            st.write(output)
        except Exception as e:
            st.error(str(e))
