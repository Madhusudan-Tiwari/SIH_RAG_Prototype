import streamlit as st
from pathlib import Path
from modules.ingestion import process_text_file, process_image_file, process_audio_file
from modules.embeddings import get_text_embedding, get_image_embedding
from modules.vector_db import VectorDB
from modules.llm_integration import query_llm_with_context

# ----------------- Configuration -----------------
SAMPLE_FOLDER = Path("sample_files")
SUPPORTED_TEXT = [".pdf", ".docx"]
SUPPORTED_IMAGE = [".png", ".jpg", ".jpeg"]
SUPPORTED_AUDIO = [".wav", ".mp3"]

st.set_page_config(page_title="🎯 Multimodal RAG Prototype", layout="wide")
st.title("🎯 Multimodal RAG Prototype")
st.caption("Upload files or test on sample files. Supports English/Hindi/Punjabi.")

# Sidebar
mode = st.sidebar.radio("Choose input mode", ["Sample Files", "Upload Files"])
top_k = st.sidebar.slider("Top-k results to retrieve", 1, 5, 3)

# ----------------- Initialize Vector DB -----------------
vector_db = VectorDB(dim=512)

# ----------------- Initialize session state -----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ----------------- Helper Function -----------------
def process_and_store(file_path):
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_TEXT:
        text = process_text_file(file_path)
        emb = get_text_embedding(text)
    elif ext in SUPPORTED_IMAGE:
        emb = get_image_embedding(file_path)
        text = f"<Image embedding stored: {file_path.name}>"
    elif ext in SUPPORTED_AUDIO:
        text = process_audio_file(file_path)
        emb = get_text_embedding(text)
    else:
        return None, None
    vector_db.add_vector(emb, text)
    return text, emb

# ----------------- Sample Files Mode -----------------
if mode == "Sample Files":
    st.subheader("Testing on sample files")
    if not SAMPLE_FOLDER.exists():
        st.warning("No sample files found in 'sample_files/' folder")
    else:
        sample_files = list(SAMPLE_FOLDER.iterdir())
        selected_file = st.selectbox("Select a sample file", sample_files)
        st.write(f"Selected: {selected_file.name}")
        content, emb = process_and_store(selected_file)
        if content:
            st.text_area("Preview / embedding info", content, height=150)
            if emb is not None:
                st.text(f"Embedding vector shape: {emb.shape}")

# ----------------- Upload Files Mode -----------------
else:
    st.subheader("Upload your own files")
    uploaded_files = st.file_uploader(
        "Upload text, image, or audio files", 
        type=[ext.strip('.') for ext in SUPPORTED_TEXT + SUPPORTED_IMAGE + SUPPORTED_AUDIO],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Processing: {file.name}")
            content, emb = process_and_store(Path(file.name))
            if content:
                st.text_area(f"Preview / embedding info: {file.name}", content, height=150)
                if emb is not None:
                    st.text(f"Embedding vector shape: {emb.shape}")

# ----------------- Chat Section -----------------
st.subheader("Chat with your data")
st.text_area("Type your question...", key="user_input", height=100)
if st.button("Send"):
    user_input = st.session_state.user_input.strip()
    if user_input:
        top_texts = vector_db.query_top_k(user_input, k=top_k)
        # Build conversation context
        conversation_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in st.session_state.conversation
        )
        answer = query_llm_with_context(user_input, top_texts, conversation_context)
        # Save messages
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": answer})
        # Clear input
        st.session_state.user_input = ""

# ----------------- Display chat history -----------------
st.subheader("Conversation History")
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align:right; background-color:#DCF8C6; padding:5px; border-radius:10px; margin:5px;'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; background-color:#F1F0F0; padding:5px; border-radius:10px; margin:5px;'>{msg['content']}</div>", unsafe_allow_html=True)
