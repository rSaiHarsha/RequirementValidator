import streamlit as st
import pandas as pd
from models import LLMManager
from rag_engine import RAGEngine
from analyzer import RequirementAnalyzer

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

# Application Layout Tabs
tab_rag, tab_analysis, tab_chat = st.tabs([
    "📂 RAG Knowledge Engine", 
    "📈 Requirements Quality Analyst", 
    "💬 Nemotron Core Chat"
])

# ==========================================
# TAB 1: RAG ENGINE CONSOLE
# ==========================================
with tab_rag:
    st.header("Knowledge Base Matrix")
    st.caption("Upload baseline reference docs, standard operations manuals, or past specifications.")
    
    uploaded_files = st.file_uploader(
        "Drop foundational files here", 
        type=["txt", "csv", "md", "log"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Train RAG Engine", use_container_width=True):
        if uploaded_files:
            st.session_state.rag.clear_database()
            for file in uploaded_files:
                st.session_state.rag.process_file(file.name, file.read())
            
            prog_bar = st.progress(0.0)
            status_box = st.empty()
            
            def update_progress(pct):
                prog_bar.progress(pct)
                status_box.text(f"Computing embeddings via NVIDIA endpoint: {int(pct * 100)}% Complete")
            
            success, msg = st.session_state.rag.train_engine(update_progress)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.warning("Please upload files before starting engine calibration.")

# ==========================================
# TAB 2: REQUIREMENT ANALYSIS INTERFACE
# ==========================================
with tab_analysis:
    st.header("INCOSE / ASPICE Automated Audit Tool")
    
    st.markdown("### 📂 Automotive V-Cycle Upload Matrix")
    st.caption("Upload specs documents corresponding to different stages of the Automotive V-Cycle.")
    
    with st.expander("🛠️ V-Cycle Level Settings", expanded=True):
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            sys2_files = st.file_uploader("SYS.2 - System Requirements Specification", type=["txt", "md", "csv"], accept_multiple_files=True, key="sys2_uploader")
            sys3_files = st.file_uploader("SYS.3 - System Architecture Design", type=["txt", "md", "csv"], accept_multiple_files=True, key="sys3_uploader")
        with col_up2:
            swe1_files = st.file_uploader("SWE.1 - Software Requirements Specification", type=["txt", "md", "csv"], accept_multiple_files=True, key="swe1_uploader")
            swe2_files = st.file_uploader("SWE.2 - Software Architectural Design", type=["txt", "md", "csv"], accept_multiple_files=True, key="swe2_uploader")
            
    if st.button("📊 Run Deep Quality Analysis"):
        # Combine uploaded files with their respective levels
        all_reqs = []
        
        def process_level_files(uploaded_files, level_label):
            reqs = []
            if uploaded_files:
                for f in uploaded_files:
                    parsed = st.session_state.analyzer.parse_requirements(f.name, f.read())
                    for text in parsed:
                        reqs.append({
                            "text": text,
                            "level": level_label,
                            "source_file": f.name
                        })
            return reqs

        all_reqs.extend(process_level_files(sys2_files, "SYS.2"))
        all_reqs.extend(process_level_files(sys3_files, "SYS.3"))
        all_reqs.extend(process_level_files(swe1_files, "SWE.1"))
        all_reqs.extend(process_level_files(swe2_files, "SWE.2"))

        if all_reqs:
            with st.spinner("Analyzing ruleset compliance using Nvidia Nemotron..."):
                rag_context = ""
                if st.session_state.rag.vectors:
                    sample_query = " ".join([r["text"] for r in all_reqs[:2]])
                    rag_context = st.session_state.rag.query(sample_query)
                    st.info("ℹ️ Augmented verification active: Using RAG context data alongside LLM foundations.")
                else:
                    st.warning("⚠️ RAG Engine offline or empty. Defaulting to native out-of-the-box LLM context.")
                
                report = st.session_state.analyzer.analyze(all_reqs, rag_context)
                
                # Metrics Rendering
                st.subheader("Executive Metrics Overview")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Compliant Requirements", report["summary"]["good_count"])
                m_col2.metric("Non-Compliant Blocks", report["summary"]["bad_count"])
                m_col3.metric("INCOSE Deficiencies", report["summary"]["failed_incose_rules"])
                m_col4.metric("ASPICE Core Violations", report["summary"]["failed_aspice_rules"])
                
                # Performance Charts
                chart_data = pd.DataFrame({
                    "Metrics": ["Compliant", "Non-Compliant", "INCOSE Flags", "ASPICE Flags"],
                    "Count": [report["summary"]["good_count"], report["summary"]["bad_count"], report["summary"]["failed_incose_rules"], report["summary"]["failed_aspice_rules"]]
                })
                st.bar_chart(chart_data, x="Metrics", y="Count")
                
                # Dataframe Logging Table
                st.subheader("Audited Requirements Log")
                df_report = pd.DataFrame(report["detailed_report"])
                st.dataframe(df_report, use_container_width=True)
                
                csv_data = df_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Compliance Corrected Report (.CSV)",
                    data=csv_data,
                    file_name="compliance_audit_report.csv",
                    mime="text/csv"
                )
        else:
            st.error("No requirement files discovered to initiate execution sequence.")

# ==========================================
# TAB 3: CORE CHATBOT INTERFACE
# ==========================================
with tab_chat:
    st.header("Interactive Engineering Terminal")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query model parameter specifications..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            response = st.session_state.llm.get_response(st.session_state.messages)
            
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        full_response += delta.content
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})