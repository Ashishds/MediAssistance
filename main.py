import streamlit as st
from app.ui import upload_pdfs
from app.pdf_utils import read_pdf_text
from app.vectorstore_utils import build_faiss_store, search_similar_docs
from app.chat_utils import load_chat_engine, query_chat_engine
from app.config import EURI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

st.set_page_config(
    page_title="MediAssist - AI Health Document Helper",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .chat-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-box.user {
        background-color: #2d3242;
        color: #fff;
    }
    .chat-box.bot {
        background-color: #f7f7f7;
        color: #000;
    }
    .chat-box .time {
        font-size: 0.75rem;
        opacity: 0.65;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_store" not in st.session_state:
    st.session_state.db_store = None
if "llm_engine" not in st.session_state:
    st.session_state.llm_engine = None

# Header
st.markdown("""
<div style="text-align:center; padding:1.5rem 0;">
    <h1 style="color:#e63946; font-size:2.8rem;">🩺 MediAssist</h1>
    <p style="font-size:1.1rem; color:#444;">AI-powered Medical Document Chatbot</p>
</div>
""", unsafe_allow_html=True)

# Sidebar uploader
with st.sidebar:
    st.subheader("📂 Upload Medical Files")
    uploaded_docs = upload_pdfs()

    if uploaded_docs:
        st.success(f"✅ {len(uploaded_docs)} file(s) added")
        
        if st.button("🔧 Process Files"):
            with st.spinner("Analyzing your documents..."):
                all_texts = [read_pdf_text(f) for f in uploaded_docs]

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )

                doc_chunks = []
                for txt in all_texts:
                    doc_chunks.extend(splitter.split_text(txt))

                store = build_faiss_store(doc_chunks)
                st.session_state.db_store = store

                engine = load_chat_engine(EURI_API_KEY)
                st.session_state.llm_engine = engine

                st.success("🎉 Processing complete!")
                st.balloons()

# Chat UI
st.markdown("### 💬 Talk to Your Files")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        st.caption(msg["time"])

if user_q := st.chat_input("Type your question about the uploaded documents..."):
    timestamp = time.strftime("%H:%M")
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_q,
        "time": timestamp
    })

    with st.chat_message("user"):
        st.markdown(user_q)
        st.caption(timestamp)

    if st.session_state.db_store and st.session_state.llm_engine:
        with st.chat_message("assistant"):
            with st.spinner("📖 Reading through documents..."):
                matches = search_similar_docs(st.session_state.db_store, user_q)

                context_data = "\n\n".join([doc.page_content for doc in matches])

                crafted_prompt = f"""
                You are MediAssist, a helpful AI assistant specialized in medical documents. 
                Use the given content to respond clearly and accurately. 
                If the answer is not present, politely mention that.

                Document Data:
                {context_data}

                Question: {user_q}

                Reply:
                """

                output = query_chat_engine(st.session_state.llm_engine, crafted_prompt)

            st.markdown(output)
            st.caption(timestamp)

            st.session_state.chat_history.append({
                "role": "assistant",
                "text": output,
                "time": timestamp
            })
    else:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process files first!")
            st.caption(timestamp)

st.markdown("---")
st.markdown(
    '<div style="text-align:center; font-size:0.9rem; color:#666;">⚡ Powered by Euri AI & LangChain</div>',
    unsafe_allow_html=True
)
