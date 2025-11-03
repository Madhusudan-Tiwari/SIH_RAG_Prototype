import streamlit as st
from pathlib import Path
import tempfile # For robust file upload handling
from modules.ingestion import process_text_file, process_image_file, process_audio_file
# Importing corrected embeddings and the new dimension
from modules.embeddings import get_text_embedding, get_image_embedding, TEXT_EMBED_DIM 
from modules.vector_db import VectorDB
from modules.llm_integration import query_llm_with_context

# ----------------- Configuration -----------------
SAMPLE_FOLDER = Path("sample_files")
SUPPORTED_TEXT = [".pdf", ".docx"]
SUPPORTED_IMAGE = [".png", ".jpg", ".jpeg"]
SUPPORTED_AUDIO = [".wav", ".mp3"]

# ----------------- Streamlit Setup -----------------
st.set_page_config(page_title="🎯 Multimodal RAG Prototype", layout="wide")
st.title("🎯 Multimodal RAG Prototype")
st.caption("Upload files or test on sample files. Supports English/Hindi/Punjabi.")

# Sidebar
mode = st.sidebar.radio("Choose input mode", ["Sample Files", "Upload Files"])
top_k = st.sidebar.slider("Top-k results to retrieve", 1, 5, 3)

# ----------------- Initialize Vector DB (FIXED for persistence) -----------------
# Store the VectorDB object in session_state to ensure persistence across reruns.
if "vector_db" not in st.session_state:
    st.session_state.vector_db = VectorDB(dim=TEXT_EMBED_DIM) 
vector_db = st.session_state.vector_db # Use the persistent instance

# ----------------- Initialize session state -----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # stores last 5 messages
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # cache embeddings to avoid recomputation

# ----------------- Helper Function -----------------
def process_and_store(file):
    # Use the persistent vector_db instance
    current_db = st.session_state.vector_db
    
    is_path = isinstance(file, Path)
    file_path = file if is_path else None
    
    # CRITICAL FIX for UploadedFile objects (not paths)
    if not is_path: 
        # Save the uploaded file to a temporary file on disk for ingestion libraries
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = Path(tmp_file.name)
        
    cache_key = str(file_path)
    if cache_key in st.session_state.processed_files:
        return st.session_state.processed_files[cache_key]

    ext = file_path.suffix.lower()
    
    # Process and get text/embedding based on file type
    if ext in SUPPORTED_TEXT:
        text = process_text_file(file_path)
        emb = get_text_embedding(text)
    elif ext in SUPPORTED_IMAGE:
        emb = get_image_embedding(file_path) 
        text = f"<Image embedding stored for: {file_path.name}>" 
    elif ext in SUPPORTED_AUDIO:
        text = process_audio_file(file_path)
        emb = get_text_embedding(text)
    else:
        return None, None

    if not text.strip():
        st.warning(f"Could not extract text from {file_path.name}. Skipping vector storage.")
        return None, None

    current_db.add_vector(emb, text)
    st.session_state.processed_files[cache_key] = (text, emb)
    return text, emb

# ----------------- Sample Files Mode -----------------
if mode == "Sample Files":
    st.subheader("Testing on sample files")
    if not SAMPLE_FOLDER.exists():
        st.warning("No sample files found in 'sample_files/' folder")
    else:
        sample_files = list(SAMPLE_FOLDER.iterdir())
        selected_file = st.selectbox("Select a sample file", sample_files)
        if selected_file:
            st.write(f"Selected: {selected_file.name}")
            content, emb = process_and_store(selected_file)
            if content is not None:
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
            content, emb = process_and_store(file) 
            if content is not None:
                st.text_area(f"Preview / embedding info: {file.name}", content, height=150)
                if emb is not None:
                    st.text(f"Embedding vector shape: {emb.shape}")

# ----------------- Chat Section (FIXED: Display Order) -----------------
st.subheader("Chat with your data")
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Type your question...", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Check if any documents have been loaded before querying
    if len(vector_db.vectors) == 0:
        st.error("Please load at least one document (Sample or Upload) before chatting.")
    else:
        # Compute embedding for query
        query_vec = get_text_embedding(user_input)

        # Retrieve top-k relevant contexts
        top_texts = vector_db.query_top_k_embedding(query_vec, k=top_k)

        # Query Gemini with context + conversation
        answer = query_llm_with_context(user_input, top_texts, st.session_state.conversation)

        # Save chat
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": answer})
        st.session_state.conversation = st.session_state.conversation[-5:]

    # Refresh the app to show new messages (or error message)
    st.rerun()

# ----------------- Display chat history (FIXED: Display Order) -----------------
st.subheader("Conversation History")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; color:#000000; background-color:#DCF8C6; "
                f"padding:8px 12px; border-radius:15px; margin:5px 0 5px auto; "
                f"max-width:80%; width:fit-content; border-bottom-right-radius:2px;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; color:#000000; background-color:#F1F0F0; "
                f"padding:8px 12px; border-radius:15px; margin:5px auto 5px 0; "
                f"max-width:80%; width:fit-content; border-bottom-left-radius:2px;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
