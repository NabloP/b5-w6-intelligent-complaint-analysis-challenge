# ==============================================================================
# app_streamlit.py — CrediTrust Complaint AI
# Polished conversational AI for complaint analysis with streaming, memory,
# markdown answers, and minimal UI inspired by Gemini and ChatGPT Plus.
# Author: Nabil Mohamed | July 2025
# ==============================================================================

# ---------------------------
# 📦 Imports
# ---------------------------
import os
import sys
import time
import pathlib
import pandas as pd
import streamlit as st

from src.rag.chroma_loader import ChromaLoader
from src.rag.retriever import QuestionRetriever
from src.rag.prompt_template import PromptEngineer
from src.rag.answer_generator import AnswerGenerator


# ---------------------------
# 🚀 Launch App
# ---------------------------
def launch_app():
    # -------------------- Project Root Detection --------------------
    current_dir = pathlib.Path(__file__).resolve().parent
    for parent in current_dir.parents:
        if (parent / "src").exists() and (parent / "vector_store").exists():
            project_root = parent
            break
    else:
        st.error("❌ Project root not found.")
        st.stop()

    sys.path.insert(0, str(project_root))
    st.set_page_config(page_title="CrediTrust Complaint AI", layout="wide")

    # -------------------- Persistent Session State --------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Holds chat messages
    if "history" not in st.session_state:
        st.session_state.history = []  # Holds exportable history
    if "theme" not in st.session_state:
        st.session_state.theme = "🌞 Light"  # Default theme
    if "ai_pending" not in st.session_state:
        st.session_state.ai_pending = False  # Tracks if AI needs to respond
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""  # Stores last question

    # -------------------- Sidebar: Controls --------------------
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        theme = st.radio(
            "Theme",
            ["🌞 Light", "🌙 Dark"],
            index=0 if st.session_state.theme == "🌞 Light" else 1,
        )
        st.session_state.theme = theme
        top_k = st.slider("Top-K Chunks", 1, 20, 5)

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.ai_pending = False
            st.session_state.last_user_input = ""
            st.rerun()

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.download_button(
                "⬇️ Export Chat", df.to_csv(index=False), file_name="chat_history.csv"
            )

    # -------------------- Theme Styling --------------------
    is_dark = st.session_state.theme == "🌙 Dark"
    bg_color = "#0d1117" if is_dark else "#ffffff"
    user_bubble = "#1e1e1e" if is_dark else "#f5f5f5"
    text_color = "#dddddd" if is_dark else "#333333"
    # Fix: Explicit user bubble text color for readability on light and dark modes
    user_text_color = "#dddddd" if is_dark else "#222222"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: {bg_color};
            color: {text_color};
        }}
        .user-bubble {{
            padding: 16px 20px;
            border-radius: 12px;
            margin: 10px auto;
            max-width: 750px;
            background-color: {user_bubble};
            color: {user_text_color};
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .typing {{
            font-size: 16px;
            animation: blink 1s infinite;
        }}
        @keyframes blink {{
            0% {{ opacity: 0.2; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.2; }}
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )

    # -------------------- Load RAG Components --------------------
    @st.cache_resource(show_spinner="Loading AI components...")
    def load_components():
        loader = ChromaLoader(
            persist_directory=str(project_root / "vector_store" / "chroma_db"),
            collection_name="complaint_chunks",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        vector_store = loader.load_collection()
        retriever = QuestionRetriever(
            vector_store, "sentence-transformers/all-MiniLM-L6-v2", top_k=top_k
        )
        prompt_engineer = PromptEngineer(max_context_length=3000)
        generator = AnswerGenerator(
            api_key=os.getenv(
                "GEMINI_API_KEY", "AIzaSyDM6YAwR1ajk6fzhGepD0q7sOtVqrnBuMs"
            ),
            model="gemini-2.5-flash",
        )
        return retriever, prompt_engineer, generator

    retriever, prompt_engineer, answer_generator = load_components()

    # -------------------- Header --------------------
    st.markdown(
        """
        <div style="padding:12px 0; text-align:center;">
            <h1 style="font-weight:600; margin-bottom:4px;">CrediTrust Complaint AI</h1>
            <p style="font-size:13px; margin-top:0;">Evidence-backed answers for consumer complaints</p>
            <hr style="border-color:#ddd; margin-top:10px;" />
        </div>
    """,
        unsafe_allow_html=True,
    )

    # -------------------- Display Chat --------------------
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-bubble">{msg['content']}</div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])  # AI message in Markdown
            if msg.get("sources"):
                with st.expander("See Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {i}:** {src}", unsafe_allow_html=True)

    # -------------------- User Input --------------------
    user_input = st.chat_input("Ask about customer complaints...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.last_user_input = user_input
        st.session_state.ai_pending = True
        st.rerun()

    # -------------------- AI Response Logic --------------------
    if st.session_state.ai_pending:
        user_question = st.session_state.last_user_input
        history = "\n".join(
            [
                f"{'Q' if m['role']=='user' else 'A'}: {m['content']}"
                for m in st.session_state.messages[-6:]
            ]
        )
        retrieved_chunks = retriever.retrieve(user_question)
        sources = [
            chunk["document"][:300].strip() + "..." for chunk in retrieved_chunks[:2]
        ]
        retrieved_context = "\n---\n".join(sources)

        full_prompt = f"""
You are a helpful complaint analysis AI. Use retrieved complaint evidence and minimal general knowledge.
Conversation:
{history}

Question:
{user_question}

Evidence:
{retrieved_context}

Answer:
"""

        response_placeholder = st.empty()
        displayed = ""
        full_answer = answer_generator.generate_answer(full_prompt)

        for char in full_answer:
            displayed += char
            response_placeholder.markdown(displayed + " ▌")
            time.sleep(0.015)

        response_placeholder.markdown(displayed)

        st.session_state.messages.append(
            {"role": "assistant", "content": displayed, "sources": sources}
        )
        st.session_state.history.append(
            {
                "Question": user_question,
                "Answer": displayed,
                "Sources": "" | ".join(sources),
            }
        )
        st.session_state.ai_pending = False

        with st.expander("See Sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {src}", unsafe_allow_html=True)


# ===============================================================================
