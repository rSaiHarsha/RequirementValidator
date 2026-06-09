import streamlit as st
from Model.llm import LLMManager
from RagEngine.rag_engine import RAGEngine
from Analysis.analyzer import RequirementAnalyzer
from UI.rag_tab import render_rag_tab
from UI.analysis_tab import render_analysis_tab
from UI.chat_tab import render_chat_tab

st.set_page_config(page_title="NVIDIA Mission Critical Assistant", layout="wide")

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