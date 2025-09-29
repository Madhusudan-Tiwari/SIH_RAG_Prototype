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
            {"name": "perplexity", "func": self.query_perplexity},
        ]

    # ----------------- OpenAI -----------------
    def query_openai(self, prompt):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # ✅ free $5 credit applies to this
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[OpenAI failed: {e}]"

# Gemini
def query_gemini(self, prompt):
    if genai is None:
        return "[Gemini failed: google-genai not installed]"
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt]  # wrap in list
        )
        return response.text if isinstance(response.text, str) else "\n".join(response.text)
    except Exception as e:
        return f"[Gemini failed: {e}]"

# Perplexity
def query_perplexity(self, prompt):
    try:
        headers = {"Authorization": f"Bearer {st.secrets['PERPLEXITY_API_KEY']}"}
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.1-sonar-small-128k-chat",
                "input": prompt  # use 'input' instead of 'messages'
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("text", "[Perplexity gave no answer]")
    except Exception as e:
        return f"[Perplexity failed: {e}]"


    # ----------------- Query All -----------------
    def query(self, prompt):
        results = []
        for llm in self.llms:
            result = llm["func"](prompt)
            results.append(f"**{llm['name'].upper()}**: {result}")
        return "\n\n".join(results)


# ----------------- Streamlit UI -----------------
st.title("🔮 Multi-LLM Fallback Demo")

prompt = st.text_input("Enter your prompt:")
if st.button("Send") and prompt:
    llm = MultiLLM()
    output = llm.query(prompt)
    st.markdown(output)
