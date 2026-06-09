import streamlit as st

def render_rag_tab():
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
