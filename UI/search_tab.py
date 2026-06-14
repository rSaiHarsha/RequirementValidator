import streamlit as st

# Test cases focusing on automotive/ADAS requirements
TEST_QUESTIONS = [
    {
        "title": "Test Case 1: Lane-Keep Assist (Vague & Combined)",
        "question": (
            "Evaluate this ADAS requirement: 'The ADAS lane-keep assist system shall always keep the car in the lane "
            "and function safely at high speed.' What specific INCOSE guidelines does it violate, and how should it "
            "be rewritten?"
        )
    },
    {
        "title": "Test Case 2: Autonomous Emergency Braking (Verifiability & Ambiguity)",
        "question": (
            "How does the INCOSE guide help in formulating verifiable requirements for Autonomous Emergency Braking (AEB)? "
            "Evaluate this requirement: 'The vehicle must apply brakes immediately when an obstacle is close.'"
        )
    },
    {
        "title": "Test Case 3: Fail-safe State (Quantifiers, Tolerances & Timing)",
        "question": (
            "Evaluate this safety-critical requirement: 'Upon detecting a sensor failure, the system must immediately trigger "
            "a fail-safe state.' Which rules about timing, precision, and avoid-vague-quantifiers apply here?"
        )
    },
    {
        "title": "Test Case 4: Pedestrian Detection (Modal Verbs & Singularity)",
        "question": (
            "Evaluate this ADAS requirement: 'The front camera should detect pedestrians, and the system must brake when needed.' "
            "What rules does this violate regarding singularity, appropriate modal verbs (shall/should/must), and precision?"
        )
    }
]

# Dialog decorator for choosing collection/knowledge base
@st.dialog("Select Knowledge Base Source")
def choose_collection_dialog(query_text):
    st.write("🔍 **Requirement under evaluation:**")
    st.info(f"\"{query_text}\"")
    st.write("Please select which knowledge base source you want to retrieve compliance rules from:")
    
    # Build collection options
    db_collections = ["All Collections", "None"]
    available_cols = st.session_state.rag.get_collections()
    for col in available_cols:
        if col not in db_collections:
            db_collections.append(col)
            
    selected_kb = st.selectbox(
        "Select Knowledge Base:",
        db_collections,
        index=0,
        key="dlg_val_selectbox"
    )
    
    col_confirm, col_cancel = st.columns([1, 1])
    with col_confirm:
        if st.button("Confirm & Analyze", type="primary", key="dlg_val_confirm", use_container_width=True):
            st.session_state.query_collection = selected_kb
            st.session_state.run_analysis = True
            st.session_state.show_val_dialog = False
            st.rerun()
    with col_cancel:
        if st.button("Cancel", key="dlg_val_cancel", use_container_width=True):
            st.session_state.show_val_dialog = False
            st.rerun()

def render_search_tab():
    if "query_val" not in st.session_state:
        st.session_state.query_val = ""
    if "main_search_input" not in st.session_state:
        st.session_state.main_search_input = ""
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "show_val_dialog" not in st.session_state:
        st.session_state.show_val_dialog = False
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = None
    if "llm_analysis" not in st.session_state:
        st.session_state.llm_analysis = None
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False
    if "query_collection" not in st.session_state:
        st.session_state.query_collection = "All Collections"

    if st.session_state.get("pending_query") is not None:
        st.session_state.main_search_input = st.session_state.pending_query
        st.session_state.query_val = st.session_state.pending_query
        st.session_state.pending_query = None

    st.markdown('<div class="rag-header">Search from your Knowledge base</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-sub-header">Automotive Requirements Verification via Knowledge Retrieval</div>', unsafe_allow_html=True)

    # Search panel UI layout
    _, center_boundary_col, _ = st.columns([1, 2.5, 1])

    with center_boundary_col:
        st.markdown("<br>", unsafe_allow_html=True)
        user_query = st.text_input(
            "Enter a requirement to evaluate or query the knowledge base...",
            value=st.session_state.query_val,
            key="main_search_input_field",
            label_visibility="collapsed"
        )
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            search_clicked = st.button(
                "🚀 Analyze", 
                type="primary", 
                use_container_width=True,
                key="val_run_btn"
            )
        with btn_col2:
            clear_clicked = st.button(
                "🧹 Clear", 
                use_container_width=True,
                key="val_clear_all_btn"
            )
            
        if clear_clicked:
            st.session_state.pending_query = ""
            st.session_state.query_val = ""
            st.session_state.retrieved_chunks = None
            st.session_state.llm_analysis = None
            st.session_state.run_analysis = False
            st.rerun()

        if search_clicked:
            if user_query.strip() == "":
                st.warning("Please enter a query first.")
            else:
                st.session_state.query_val = user_query
                st.session_state.show_val_dialog = True
                st.rerun()

        # Suggestions list
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.95rem; font-weight: 500; margin-bottom: 12px;'>💡 Quick Suggestions</div>", unsafe_allow_html=True)
        
        cols = st.columns(4)
        for idx, test in enumerate(TEST_QUESTIONS):
            with cols[idx]:
                if st.button(test["title"], key=f"btn_suggestion_{idx}", use_container_width=True):
                    st.session_state.pending_query = test["question"]
                    st.session_state.show_val_dialog = True
                    st.rerun()

    # Modal call
    if st.session_state.get("show_val_dialog", False):
        choose_collection_dialog(st.session_state.query_val)

    # RAG Pipeline execution
    if st.session_state.run_analysis:
        query_collection = st.session_state.get("query_collection", "All Collections")
        
        with st.spinner("Embedding query & searching Vector DB..."):
            try:
                # Search using upgraded RAGEngine
                results = st.session_state.rag.search(
                    search_text=st.session_state.query_val,
                    collection_name=query_collection,
                    top_k=4
                )
                st.session_state.retrieved_chunks = results
                
                context_texts = []
                for idx, r in enumerate(results, 1):
                    payload = r["payload"]
                    meta = payload.get("metadata", {})
                    text = payload.get("text", "")
                    section = meta.get("section", "N/A")
                    item_id = meta.get("item_id") or "N/A"
                    item_name = meta.get("item_name") or "N/A"
                    page = meta.get("page", "N/A")
                    source_col = r["collection"]
                    context_texts.append(
                        f"Source: Collection {source_col}, Section {section}, Page {page}, Item {item_id} ({item_name})\n"
                        f"Content: {text}"
                    )
                
                context_block = "\n\n---\n\n".join(context_texts) if context_texts else "No context rules found."

                system_prompt = (
                    "You are an expert systems engineer specializing in automotive systems and ADAS. "
                    "Use the provided context from the INCOSE Guide to Writing Requirements to answer the question. "
                    "Critique the requirement under evaluation using specific rules (R##) or characteristics (C##) "
                    "mentioned in the context. Show how to write it correctly based on the INCOSE guidance. "
                    "If the context doesn't contain the specific rule, use your general engineering knowledge to apply "
                    "INCOSE-aligned reasoning, but prioritize citing rules found in the context."
                )
                user_prompt = f"Context:\n{context_block}\n\nQuestion:\n{st.session_state.query_val}"
                
                with st.spinner("Generating compliance report..."):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    completion = st.session_state.llm.get_response(messages, stream=False)
                    st.session_state.llm_analysis = completion.choices[0].message.content
                    
            except Exception as e:
                st.error(f"Error executing RAG search pipeline: {e}")
                st.session_state.retrieved_chunks = None
                st.session_state.llm_analysis = None
            finally:
                st.session_state.run_analysis = False
                st.rerun()

    # Display results
    if st.session_state.retrieved_chunks is not None:
        st.markdown("---")
        st.markdown("### 🤖 INCOSE Compliance Critique")
        if st.session_state.llm_analysis:
            st.markdown(st.session_state.llm_analysis)
        else:
            st.info("No compliance critique generated.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display retrieved chunks
        query_collection = st.session_state.get("query_collection", "All Collections")
        with st.expander(f"📚 View Retrieved Guidelines (Source: {query_collection})", expanded=False):
            if not st.session_state.retrieved_chunks:
                st.info("No guidelines retrieved (None selected or no search results).")
            else:
                for rank, r in enumerate(st.session_state.retrieved_chunks, 1):
                    payload = r["payload"]
                    meta = payload.get("metadata", {})
                    text = payload.get("text", "")
                    section = meta.get("section", "N/A")
                    item_id = meta.get("item_id") or "N/A"
                    item_name = meta.get("item_name") or "N/A"
                    page = meta.get("page", "N/A")
                    score = r["score"]
                    source_col = r["collection"]
                    item_type = meta.get("item_type", "N/A")
                    keywords = meta.get("keywords", [])
                    
                    safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
                    safe_item_name = item_name.replace("<", "&lt;").replace(">", "&gt;")
                    safe_item_id = item_id.replace("<", "&lt;").replace(">", "&gt;")
                    
                    tags_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                    
                    card_html = f"""
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <span class="chunk-badge score">#{rank} • Score {score:.4f}</span>
                            <span class="chunk-badge section">Sec: {section}</span>
                            <span class="chunk-badge item">{safe_item_id}</span>
                            <span class="chunk-badge page">Page {page}</span>
                            <span class="chunk-badge type">DB: {source_col}</span>
                            <span class="chunk-badge type">{item_type}</span>
                        </div>
                        <div class="chunk-body">
                            <strong>{safe_item_name}:</strong> {safe_text}
                        </div>
                        <div class="tag-container">
                            {tags_html}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
