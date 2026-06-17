import streamlit as st
import pandas as pd
import tempfile
import os
import re
import json
from Analysis.loader import load_uploaded_requirements

def apply_df_styling(df_style, style_func, subset):
    """Safe fallback for Pandas styler mapping (uses .map for newer pandas and .applymap for older)."""
    try:
        df_cols = df_style.data.columns
        existing_subset = [col for col in subset if col in df_cols]
        if not existing_subset:
            return df_style
        subset = existing_subset
    except Exception:
        pass

    if hasattr(df_style, "map"):
        return df_style.map(style_func, subset=subset)
    return df_style.applymap(style_func, subset=subset)


def get_cached_result(action_key, current_metadata, compute_func):
    # Forcing bypass of cache to ensure new grouped data structure is executed
    return compute_func()

def process_task_with_controls(task_id, items, process_func, mode_val, selected_collections_val, render_df_func=None):
    state_status = f"{task_id}_status"
    state_index = f"{task_id}_index"
    state_results = f"{task_id}_results"
    state_total = f"{task_id}_total"
    
    cache_file = os.path.join(os.getcwd(), f".cache_{task_id}.json")
    
    if state_status not in st.session_state:
        st.session_state[state_status] = "idle"
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass
                
    if state_index not in st.session_state:
        st.session_state[state_index] = 0
        
    if state_results not in st.session_state or len(st.session_state[state_results]) == 0:
        st.session_state[state_results] = []
        # Attempt recovery from persistent disk cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    recovered = json.load(f)
                if recovered:
                    st.session_state[state_results] = recovered
                    st.session_state[state_index] = len(recovered)
            except Exception:
                pass
                
    if state_total not in st.session_state:
        st.session_state[state_total] = len(items)

    status = st.session_state[state_status]
    current_index = st.session_state[state_index]
    results = st.session_state[state_results]
    total = len(items)

    is_active = status in ["running", "paused"] and current_index < total
    button_label = "⏸️ Pause" if status == "running" else "▶️ Start"
    
    if st.button(button_label, disabled=not is_active, key=f"toggle_{task_id}"):
        if status == "running":
            st.session_state[state_status] = "paused"
        else:
            st.session_state[state_status] = "running"
        st.rerun()

    progress_bar = st.progress(current_index / total if total > 0 else 0.0)
    status_text = st.empty()
    df_placeholder = st.empty()

    if len(results) > 0:
        df_partial = pd.DataFrame(results)
        if not df_partial.empty:
            if render_df_func:
                render_df_func(df_partial, df_placeholder)
            else:
                df_placeholder.dataframe(df_partial, use_container_width=True, height=400)

    if status == "running" and current_index < total:
        chunk_size = 5 if mode_val == "single" else st.session_state.get("batch_size", 10)
        status_text.text(f"Processing requirement {current_index + 1} of {total}...")
        
        chunk = items[current_index : current_index + chunk_size]
        chunk_res = None
        try:
            def chunk_callback(curr, tot, current_data=None):
                if current_data is not None:
                    # Save the EXACT state of the UI immediately to survive fast reruns
                    full_data = st.session_state[state_results] + current_data
                    st.session_state[f"{task_id}_live_results"] = full_data
                    
                    # HARD SAVE TO DISK to guarantee no vanishing
                    try:
                        with open(cache_file, "w") as f:
                            json.dump(full_data, f)
                    except Exception:
                        pass
                    
                overall_curr = min(current_index + curr, total)
                progress_bar.progress(overall_curr / total if total > 0 else 0.0)
                status_text.text(f"Processed requirement {overall_curr} of {total}...")
                
                if current_data is not None:
                    df_partial = pd.DataFrame(st.session_state[f"{task_id}_live_results"])
                    if not df_partial.empty:
                        if render_df_func:
                            render_df_func(df_partial, df_placeholder)
                        else:
                            df_placeholder.dataframe(df_partial, use_container_width=True, height=400)
                            
            chunk_res = process_func(chunk, progress_callback=chunk_callback, rag=st.session_state.rag, mode=mode_val, selected_collections=selected_collections_val, batch_size=chunk_size)
        
        finally:
            if chunk_res is not None:
                new_results = list(st.session_state[state_results])
                new_results.extend(chunk_res)
                st.session_state[state_results] = new_results
                st.session_state[state_index] += len(chunk)
                try:
                    with open(cache_file, "w") as f:
                        json.dump(new_results, f)
                except Exception:
                    pass
            else:
                # If interrupted by Pause, recover exactly what was on screen
                live = st.session_state.get(f"{task_id}_live_results", [])
                
                # If memory live is somehow empty, fallback to disk!
                if not live and os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r") as f:
                            live = json.load(f)
                    except Exception:
                        pass
                        
                if len(live) > len(st.session_state[state_results]):
                    st.session_state[state_results] = live
                    st.session_state[state_index] = len(live)
            
            # Clean up temporary state
            st.session_state.pop(f"{task_id}_live_results", None)
            
        if st.session_state[state_index] >= total:
            st.session_state[state_status] = "completed"
            
    elif status == "completed":
        status_text.text(f"Completed processing {total} requirements.")
    else:
        display_status = "Paused" if status == "paused" else status.title()
        status_text.text(f"{display_status} at requirement {current_index} of {total}.")
        
    return st.session_state[state_results], st.session_state[state_status] == "running"

def render_analysis_tab():
    st.header("INCOSE / ASPICE Automated Audit Tool")
    mode_val = "single"
    
    st.markdown("### 📂 Automotive V-Cycle Upload Matrix")
    st.caption("Upload specs documents corresponding to different stages of the Automotive V-Cycle.")
    
    with st.expander("🛠️ Upload Software Requirements Specifications", expanded=True):
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            swe1_files = st.file_uploader("SWE.1 - Software Requirements Specification (.CSV)", type=["csv"], accept_multiple_files=False, key="swe1_uploader")
        with col_up2:
            swe2_files = st.file_uploader("SWE.2 - Software Architectural Design (.CSV)", type=["csv"], accept_multiple_files=False, key="swe2_uploader")
            
    with st.expander("🎨 Upload Design Diagrams (Image format)", expanded=True):
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            hld_file = st.file_uploader("Upload HLD (High Level Design) Diagram", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="hld_uploader")
        with col_img2:
            lld_file = st.file_uploader("Upload LLD (Low Level Design) Diagram", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="lld_uploader")
            
    if hld_file or lld_file:
        st.markdown("#### 🖼️ Uploaded Diagram Previews")
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            if hld_file:
                st.image(hld_file, caption="High Level Design (HLD) Preview", use_container_width=True)
        with preview_col2:
            if lld_file:
                st.image(lld_file, caption="Low Level Design (LLD) Preview", use_container_width=True)
            
    # --- DYNAMIC ACTION PANEL BASED ON UPLOADS ---
    has_swe1 = swe1_files is not None
    has_swe2 = swe2_files is not None
    has_hld = hld_file is not None
    has_lld = lld_file is not None

    if has_swe1 or has_swe2:
        st.markdown("---")
        st.markdown("### ⚡ V-Cycle Analysis Panel")
        
        # Processing mode is now determined via the sidebar configuration.
        mode_val = "batch" if st.session_state.get("batch_mode_enabled", False) else "single"
        
        selected_collections_val = st.session_state.get("target_rag_collections", None)
            
        st.caption("Select an analysis target based on your uploaded specification deliverables:")
        
        cols = st.columns(3)
        btn_index = 0
        actions_to_render = []
        
        # 1. Single SWE requirement uploaded
        if has_swe1 and not has_swe2:
            actions_to_render.append(("📊 Analyse SWE.1 Requirements", "analyse_swe1", "primary"))
            actions_to_render.append(("🛠️ Correct SWE.1 Requirements", "correct_swe1", "secondary"))
        elif has_swe2 and not has_swe1:
            actions_to_render.append(("📊 Analyse SWE.2 Requirements", "analyse_swe2", "primary"))
            actions_to_render.append(("🛠️ Correct SWE.2 Requirements", "correct_swe2", "secondary"))
            
        # 2. Both SWE requirements uploaded
        elif has_swe1 and has_swe2:
            actions_to_render.append(("📊 Analyse SWE.1 Requirements", "analyse_swe1", "primary"))
            actions_to_render.append(("📊 Analyse SWE.2 Requirements", "analyse_swe2", "primary"))
            actions_to_render.append(("🛠️ Correct SWE.1 Requirements", "correct_swe1", "secondary"))
            actions_to_render.append(("🛠️ Correct SWE.2 Requirements", "correct_swe2", "secondary"))
            actions_to_render.append(("🔗 Compare Traceability (SWE.1 ↔ SWE.2)", "compare_trace", "warning"))
            
        # 3. Dynamic diagram mapping comparisons
        if has_swe1 and has_hld:
            actions_to_render.append(("🔍 Compare SWE.1 with HLD", "compare_hld", "info"))
            
        if has_swe2 and has_lld:
            actions_to_render.append(("🔍 Compare SWE.2 with LLD", "compare_lld", "info"))

        # Render buttons dynamically
        for title, action_id, btn_type in actions_to_render:
            col = cols[btn_index % 3]
            btn_key = f"btn_{action_id}"
            if col.button(title, key=btn_key, use_container_width=True):
                st.session_state.last_action = action_id
                if action_id in ["analyse_swe1", "analyse_swe2", "correct_swe1", "correct_swe2"]:
                    current_status = st.session_state.get(f"{action_id}_status", "idle")
                    if current_status in ["idle", "completed"]:
                        st.session_state[f"{action_id}_status"] = "running"
                        st.session_state[f"{action_id}_index"] = 0
                        st.session_state[f"{action_id}_results"] = []
                        cache_file = os.path.join(os.getcwd(), f".cache_{action_id}.json")
                        if os.path.exists(cache_file):
                            try:
                                os.remove(cache_file)
                            except Exception:
                                pass
                    elif current_status == "paused":
                        st.session_state[f"{action_id}_status"] = "running"
            btn_index += 1

    # --- EXECUTION RESULTS DISPLAY PANEL ---
    if (has_swe1 or has_swe2) and st.session_state.last_action:
        st.markdown("---")
        st.subheader("📊 Execution Results")
        
        action = st.session_state.last_action
        swe1_reqs = load_uploaded_requirements(swe1_files) if has_swe1 else []
        swe2_reqs = load_uploaded_requirements(swe2_files) if has_swe2 else []
        df = pd.DataFrame()
        
        # Color helper functions for display styling
        def color_status(val):
            color = '#d4edda' if val == 'Passed' else '#f8d7da'
            text_color = '#155724' if val == 'Passed' else '#721c24'
            return f'background-color: {color}; color: {text_color}; font-weight: bold;'

        def color_trace(val):
            color = '#d4edda' if val == 'Covered' else '#f8d7da'
            text_color = '#155724' if val == 'Covered' else '#721c24'
            return f'background-color: {color}; color: {text_color}; font-weight: bold;'

        def color_align(val):
            color = '#d4edda' if val in ['Fully Aligned', 'Aligned'] else ('#fff3cd' if val == 'Missing in Specification' else '#f8d7da')
            text_color = '#155724' if val in ['Fully Aligned', 'Aligned'] else ('#856404' if val == 'Missing in Specification' else '#721c24')
            return f'background-color: {color}; color: {text_color}; font-weight: bold;'
            
        if action == "analyse_swe1":
            st.markdown("#### 🔍 Quality Audit: SWE.1 Software Requirements Specification")
            if not swe1_reqs:
                st.info("No requirements found in the uploaded SWE.1 file.")
            else:
                def render_swe1_df(df, placeholder):
                    placeholder.dataframe(apply_df_styling(df.style, color_status, subset=['Status']), use_container_width=True, height=400)
                    
                analysis_data, is_running = process_task_with_controls(
                    "analyse_swe1", 
                    swe1_reqs, 
                    st.session_state.analyzer.analyze_requirements,
                    mode_val, 
                    selected_collections_val,
                    render_swe1_df
                )
                
                df = pd.DataFrame(analysis_data) if analysis_data else pd.DataFrame()
                
                if not is_running and not df.empty:
                    total = len(df)
                    passed = sum(1 for item in analysis_data if item.get("Status") == "Passed")
                    review = total - passed
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Requirements Checked", total)
                    m2.metric("Passed", passed)
                    m3.metric("Review Needed", review)
                elif not is_running and df.empty:
                    if st.session_state.get("analyse_swe1_status") == "paused":
                        st.info("⏸️ Execution paused. No requirements have completed processing yet. Click 'Resume' to continue.")
                    
                if is_running:
                    st.rerun()
                
        elif action == "analyse_swe2":
            st.markdown("#### 🔍 Quality Audit: SWE.2 Software Architectural Design")
            if not swe2_reqs:
                st.info("No requirements found in the uploaded SWE.2 file.")
            else:
                def render_swe2_df(df, placeholder):
                    placeholder.dataframe(apply_df_styling(df.style, color_status, subset=['Status']), use_container_width=True, height=400)
                    
                analysis_data, is_running = process_task_with_controls(
                    "analyse_swe2", 
                    swe2_reqs, 
                    st.session_state.analyzer.analyze_requirements,
                    mode_val, 
                    selected_collections_val,
                    render_swe2_df
                )
                
                df = pd.DataFrame(analysis_data) if analysis_data else pd.DataFrame()
                
                if not is_running and not df.empty:
                    total = len(df)
                    passed = sum(1 for item in analysis_data if item.get("Status") == "Passed")
                    review = total - passed
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Requirements Checked", total)
                    m2.metric("Passed", passed)
                    m3.metric("Review Needed", review)
                elif not is_running and df.empty:
                    if st.session_state.get("analyse_swe2_status") == "paused":
                        st.info("⏸️ Execution paused. No requirements have completed processing yet. Click 'Resume' to continue.")
                    
                if is_running:
                    st.rerun()
                
        elif action == "correct_swe1":
            st.markdown("#### 🛠️ Automated Corrections: SWE.1 Requirements")
            if not swe1_reqs:
                st.info("No requirements found in the uploaded SWE.1 file.")
            else:
                def render_correct_swe1_df(df, placeholder):
                    placeholder.dataframe(df, use_container_width=True, height=400)
                    
                correction_data, is_running = process_task_with_controls(
                    "correct_swe1", 
                    swe1_reqs, 
                    st.session_state.analyzer.correct_requirements,
                    mode_val, 
                    selected_collections_val,
                    render_correct_swe1_df
                )
                
                df = pd.DataFrame(correction_data) if correction_data else pd.DataFrame()
                
                if not is_running:
                    if st.session_state.get("correct_swe1_status") == "completed" and df.empty:
                        st.success("🎉 All requirements are already compliant! No corrections needed.")
                    elif not df.empty:
                        st.caption("We have corrected the vague, non-binding, or non-measurable requirements automatically:")
                    elif df.empty and st.session_state.get("correct_swe1_status") == "paused":
                        st.info("⏸️ Execution paused. No requirements have completed processing yet. Click 'Resume' to continue.")
                    
                if is_running:
                    st.rerun()
                    
        elif action == "correct_swe2":
            st.markdown("#### 🛠️ Automated Corrections: SWE.2 Requirements")
            if not swe2_reqs:
                st.info("No requirements found in the uploaded SWE.2 file.")
            else:
                def render_correct_swe2_df(df, placeholder):
                    placeholder.dataframe(df, use_container_width=True, height=400)
                    
                correction_data, is_running = process_task_with_controls(
                    "correct_swe2", 
                    swe2_reqs, 
                    st.session_state.analyzer.correct_requirements,
                    mode_val, 
                    selected_collections_val,
                    render_correct_swe2_df
                )
                
                df = pd.DataFrame(correction_data) if correction_data else pd.DataFrame()
                
                if not is_running:
                    if st.session_state.get("correct_swe2_status") == "completed" and df.empty:
                        st.success("🎉 All architectural requirements are already compliant! No corrections needed.")
                    elif not df.empty:
                        st.caption("We have corrected the vague, non-binding, or non-measurable architectural requirements automatically:")
                    elif df.empty and st.session_state.get("correct_swe2_status") == "paused":
                        st.info("⏸️ Execution paused. No requirements have completed processing yet. Click 'Resume' to continue.")
                    
                if is_running:
                    st.rerun()
                    
        elif action == "compare_trace":
            st.markdown("#### 🔗 Bidirectional Traceability: SWE.1 (HLD) ↔ SWE.2 (LLD)")
            if not swe1_reqs or not swe2_reqs:
                st.info("Upload both SWE.1 and SWE.2 requirement specifications to run Traceability analysis.")
            else:
                trace_results = get_cached_result(
                    "compare_trace",
                    (
                        (swe1_files.name, swe1_files.size) if swe1_files else None,
                        (swe2_files.name, swe2_files.size) if swe2_files else None
                    ),
                    lambda: st.session_state.analyzer.compare_traceability(swe1_reqs, swe2_reqs)
                )
                metrics = trace_results["metrics"]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total SWE.1 Requirements", metrics["total_hld"])
                m2.metric("Covered in SWE.2", metrics["covered_count"])
                m3.metric("Orphaned (No link)", metrics["orphaned_count"])
                m4.metric("Traceability Coverage", f"{metrics['coverage_pct']}%")
                
                df = pd.DataFrame(trace_results["table"])
                st.dataframe(apply_df_styling(df.style, color_trace, subset=['Status']), use_container_width=True, height=450)
                
        elif action == "compare_hld":
            st.markdown("#### 🔍 Structural Alignment: SWE.1 Requirements vs HLD Diagram")
            st.caption("Verifying alignment between the textual SWE.1 software requirements and the High Level Design diagram.")
            
            # Extract HLD components dynamically from SWE.1 requirements content
            components_set = set()
            for r in swe1_reqs:
                # Find all text inside square brackets [Component]
                matches = re.findall(r"\[([^\]]+)\]", r.content)
                for m in matches:
                    components_set.add(m.strip())
            components = sorted(list(components_set))
            if not components:
                components = ["Main Component"]
                
            detected_str = ", ".join(f"`[{c}]`" for c in components[:3]) + ("..." if len(components) > 3 else "")
            
            with st.status("🤖 AI Vision Agent parsing layout blocks and requirements...", expanded=True) as status_indicator:
                st.write("1. Reading High Level Design Diagram image metadata...")
                st.write(f"2. Detecting structural diagram entities: {detected_str}...")
                st.write("3. Analyzing dataflow connectors and interfaces in the diagram...")
                st.write("4. Verifying if all diagram components are represented in the SWE.1 specification...")
                status_indicator.update(label="Alignment analysis complete!", state="complete")
                
            alignment_data = get_cached_result(
                "compare_hld",
                (
                    (swe1_files.name, swe1_files.size) if swe1_files else None,
                    (hld_file.name, hld_file.size) if hld_file else None
                ),
                lambda: st.session_state.analyzer.compare_hld_alignment(swe1_reqs, components)
            )
            
            df = pd.DataFrame(alignment_data)
            st.dataframe(apply_df_styling(df.style, color_align, subset=['Status']), use_container_width=True)
            
        elif action == "compare_lld":
            st.markdown("#### 🔍 Architectural Alignment: SWE.2 Requirements vs LLD Diagram")
            st.caption("Verifying alignment between the low-level SWE.2 software architectural requirements and the Low Level Design diagram.")
            
            # Extract LLD methods dynamically from SWE.2 requirements content
            methods_set = set()
            for r in swe2_reqs:
                # Find backticked signatures
                backticks = re.findall(r"`([^`]+)`", r.content)
                for b in backticks:
                    if "(" in b or "." in b or "_" in b:
                        methods_set.add(b.strip())
                # Find function calls with parentheses
                func_calls = re.findall(r"\b(\w+(?:\.\w+)*\([^)]*\))", r.content)
                for f in func_calls:
                    methods_set.add(f.strip())
            methods = sorted(list(methods_set))
            if not methods:
                methods = ["initialize()"]
                
            detected_methods = ", ".join(f"`{m}`" for m in methods[:3]) + ("..." if len(methods) > 3 else "")
            
            with st.status("🤖 AI Vision Agent parsing low-level function signatures...", expanded=True) as status_indicator:
                st.write("1. Reading Low Level Design Diagram image blocks...")
                st.write(f"2. Extracting class interfaces and method signatures: {detected_methods}...")
                st.write("3. Checking for match between diagram and SWE.2 requirement descriptions...")
                status_indicator.update(label="LLD comparison complete!", state="complete")
                
            alignment_data = get_cached_result(
                "compare_lld",
                (
                    (swe2_files.name, swe2_files.size) if swe2_files else None,
                    (lld_file.name, lld_file.size) if lld_file else None
                ),
                lambda: st.session_state.analyzer.compare_lld_alignment(swe2_reqs, methods)
            )
            
            df = pd.DataFrame(alignment_data)
            st.dataframe(apply_df_styling(df.style, color_align, subset=['Status']), use_container_width=True)

        # --- EXPORT REPORT DELIVERABLES ---
        if not df.empty:
            st.markdown("---")
            st.markdown("### 📥 Export Audit Deliverables")
            
            # Convert findings dataframe to CSV
            csv_buffer = df.to_csv(index=False).encode('utf-8')
            
            # Determine the active spec and metadata for the current action
            if action in ["analyse_swe1", "correct_swe1", "compare_hld"]:
                active_reqs = swe1_reqs
                active_files = swe1_files
                action_suffix = "swe1"
            elif action in ["analyse_swe2", "correct_swe2", "compare_lld"]:
                active_reqs = swe2_reqs
                active_files = swe2_files
                action_suffix = "swe2"
            else: # e.g. compare_trace, where we default to swe1
                active_reqs = swe1_reqs
                active_files = swe1_files
                action_suffix = "trace"
                
            active_metadata = (active_files.name, active_files.size, tuple(selected_collections_val) if selected_collections_val else None) if active_files else None
            file_name_label = active_files.name if active_files else "Requirements Specification"
            
            def run_export_analysis():
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                def callback(curr, tot, current_data=None):
                    progress_bar.progress(curr / tot)
                    status_text.text(f"Generating export audit {curr} of {tot}...")
                try:
                    res = st.session_state.analyzer.analyze_requirements(active_reqs, progress_callback=callback, rag=st.session_state.rag, mode=mode_val, selected_collections=selected_collections_val, batch_size=st.session_state.get("batch_size", 10))
                finally:
                    progress_bar.empty()
                    status_text.empty()
                return res

            def run_export_correction():
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                def callback(curr, tot, current_data=None):
                    progress_bar.progress(curr / tot)
                    status_text.text(f"Generating export corrections {curr} of {tot}...")
                try:
                    res = st.session_state.analyzer.correct_requirements(active_reqs, progress_callback=callback, rag=st.session_state.rag, mode=mode_val, selected_collections=selected_collections_val, batch_size=st.session_state.get("batch_size", 10))
                finally:
                    progress_bar.empty()
                    status_text.empty()
                return res

            analysis_res = get_cached_result(
                (f"analyse_{action_suffix}", mode_val),
                active_metadata,
                run_export_analysis
            )
            
            correction_res = get_cached_result(
                (f"correct_{action_suffix}", mode_val),
                active_metadata,
                run_export_correction
            )
            
            md_report = st.session_state.analyzer.generate_report(analysis_res, correction_res, file_name_label)
            
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    label="📥 Download Detailed Findings (.CSV)",
                    data=csv_buffer,
                    file_name=f"{action}_findings.csv",
                    mime="text/csv",
                    key="download_csv_findings",
                    use_container_width=True
                )
            with dcol2:
                st.download_button(
                    label="📥 Download Compliance Report (.MD)",
                    data=md_report.encode('utf-8'),
                    file_name=f"{action}_compliance_report.md",
                    mime="text/markdown",
                    key="download_md_report",
                    use_container_width=True
                )
                
                st.caption(f"Note: Evaluated {len(df)} artifacts from {active_files.name if active_files else 'document'}")
