import streamlit as st
from Model.llm import LLMManager
from RagEngine.rag_engine import RAGEngine
from Analysis.analyzer import RequirementAnalyzer
from UI.rag_tab import render_rag_tab

from UI.analysis_tab import render_analysis_tab
from UI.chat_tab import render_chat_tab

st.set_page_config(page_title="NVIDIA Mission Critical Assistant", layout="wide", initial_sidebar_state="collapsed")

# Initialization Management
if "llm" not in st.session_state:
    st.session_state.llm = LLMManager()
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine(st.session_state.llm)
    st.session_state.rag.load_trained_engine()
if "analyzer" not in st.session_state:
    st.session_state.analyzer = RequirementAnalyzer(st.session_state.llm)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_action" not in st.session_state:
    st.session_state.last_action = None

# System Configuration Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Customize system parameters and API fault-tolerance.")
    
    llm_retry_limit = st.slider(
        "LLM Retry Limit",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Number of retries when an LLM API request fails (e.g. transient connection error, rate limits)."
    )
    st.session_state.llm_retries = llm_retry_limit
    if "llm" in st.session_state:
        st.session_state.llm.retries = llm_retry_limit
    
    st.markdown("---")
    st.subheader("🤖 Active LLM Model")
    st.info(f"**Model:**\n`{st.session_state.llm.model_name}`")
    st.caption("Powered by NVIDIA NIM Core engine.")

    st.markdown("---")
    st.subheader("📚 Collections & Documents")
    collections = st.session_state.rag.get_collections()
    if collections:
        for col in collections:
            with st.expander(f"📁 {col}"):
                # Get documents for this collection
                col_docs = [doc for doc in st.session_state.rag.documents if doc.get("collection") == col]
                if col_docs:
                    # Try to extract unique sources or names
                    sources = set()
                    for doc in col_docs:
                        src = doc.get("source", "Unknown Source")
                        # Try to strip block/page info to get the root document name if possible
                        # e.g. "file.pdf (Block 1)" -> "file.pdf"
                        if " (Block " in src:
                            src = src.split(" (Block ")[0]
                        elif src.startswith("Page "):
                            # For Qdrant chunks without filenames, we can use title
                            title = doc.get("title", "Untitled")
                            src = f"{title} (Page {src.split(' ')[1]})"
                        sources.add(src)
                    
                    for src in sorted(list(sources)):
                        st.markdown(f"- 📄 {src}")
                else:
                    st.caption("No documents locally tracked.")
    else:
        st.info("No collections found.")

# Application Layout Tabs
tab_rag, tab_analysis, tab_chat = st.tabs([
    "📂 RAG Knowledge Engine", 
    "📈 Requirements Quality Analyst", 
    "💬 Nemotron Core Chat"
])

with tab_rag:
    render_rag_tab()

with tab_analysis:
    render_analysis_tab()

with tab_chat:
    render_chat_tab()