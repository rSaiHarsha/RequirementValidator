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


def color_status(val):
    color = '#d4edda' if val == 'Passed' else '#f8d7da'
    text_color = '#155724' if val == 'Passed' else '#721c24'
    return f'background-color: {color}; color: {text_color}; font-weight: bold;'


def get_history_limit() -> int:
    """Read the history limit from st.session_state, defaulting to 5."""
    try:
        if "history_max_items" in st.session_state:
            return max(2, int(st.session_state.history_max_items))
    except Exception:
        pass
    return 5


def add_to_history(action_id: str, results: list, source_name: str):
    """Add a completed run's results to the session state requirement history."""
    if "requirement_history" not in st.session_state:
        st.session_state.requirement_history = []
        
    added_key = f"{action_id}_added_to_history"
    if st.session_state.get(added_key):
        return
        
    if not results:
        return
        
    import uuid
    from datetime import datetime
    
    action_type = "Analyze" if "analyse" in action_id else "Correct"
    
    history_item = {
        "id": str(uuid.uuid4()),
        "type": action_type,
        "action_id": action_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requirement": f"{action_id.replace('_', ' ').title()}: {source_name}",
        "result": results,
        "source_file": source_name
    }
    
    # Prepend to history (newest first)
    st.session_state.requirement_history.insert(0, history_item)
    
    # Mark as added
    st.session_state[added_key] = True
    
    # Auto-cleanup based on config limit
    limit = get_history_limit()
    if len(st.session_state.requirement_history) > limit:
        st.session_state.requirement_history = st.session_state.requirement_history[:limit]

    # Reset task state to idle so it doesn't show in the active panel anymore
    st.session_state[f"{action_id}_status"] = "idle"
    st.session_state[f"{action_id}_results"] = []
    st.session_state.pop(f"{action_id}_index", None)
    st.session_state.pop(f"{action_id}_live_results", None)
    
    # Clear cache file to save space
    cache_file = os.path.join(os.getcwd(), f".cache_{action_id}.json")
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
        except Exception:
            pass
            
    # Trigger a rerun so the new history item shows up immediately
    st.rerun()


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
        
    if state_results not in st.session_state:
        st.session_state[state_results] = []
        
    # Proactive recovery to prevent race conditions on fast pause/resume
    curr_len = len(st.session_state[state_results])
    live = st.session_state.get(f"{task_id}_live_results", [])
    if live and len(live) > curr_len:
        st.session_state[state_results] = live
        st.session_state[state_index] = len(live)
        curr_len = len(live)
        
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                disk_live = json.load(f)
            if disk_live and len(disk_live) > curr_len:
                st.session_state[state_results] = disk_live
                st.session_state[state_index] = len(disk_live)
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
    
    if status in ["running", "paused"]:
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
            
        if st.session_state[state_index] >= total:
            st.session_state[state_status] = "completed"
            st.rerun()
            
    elif status == "completed":
        status_text.text(f"Completed processing {total} requirements.")
    else:
        display_status = "Paused" if status == "paused" else status.title()
        status_text.text(f"{display_status} at requirement {current_index} of {total}.")
        
    return st.session_state[state_results], st.session_state[state_status] == "running"

def render_analysis_tab():
    st.header("INCOSE / ASPICE Automated Audit Tool")
    mode_val = "single"
    should_rerun = False
    
    if "requirement_history" not in st.session_state:
        st.session_state.requirement_history = []
    
    st.markdown("### 📂 Automotive V-Cycle Upload Matrix")
    st.caption("Upload specs documents corresponding to different stages of the Automotive V-Cycle.")
    
    with st.expander("🛠️ Upload Software Requirements Specifications", expanded=True):
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            swe1_files = st.file_uploader("SWE.1 - Software Requirements Specification (.CSV)", type=["csv"], accept_multiple_files=False, key="swe1_uploader")
        with col_up2:
            swe2_files = st.file_uploader("SWE.2 - Software Architectural Design (.CSV)", type=["csv"], accept_multiple_files=False, key="swe2_uploader")
            
    # Check completed tasks and add to history
    for task_action in ["analyse_swe1", "analyse_swe2", "correct_swe1", "correct_swe2"]:
        status_key = f"{task_action}_status"
        if st.session_state.get(status_key) == "completed":
            results_key = f"{task_action}_results"
            completed_results = st.session_state.get(results_key, [])
            if completed_results:
                source_name = swe1_files.name if "swe1" in task_action and swe1_files else (swe2_files.name if "swe2" in task_action and swe2_files else "Requirements Specification")
                add_to_history(task_action, completed_results, source_name)
                
    # --- DYNAMIC ACTION PANEL BASED ON UPLOADS ---
    has_swe1 = swe1_files is not None
    has_swe2 = swe2_files is not None

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
                        st.session_state.pop(f"{action_id}_live_results", None)
                        st.session_state.pop(f"{action_id}_added_to_history", None)
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
    action = st.session_state.last_action
    active_status = st.session_state.get(f"{action}_status", "idle") if action else "idle"
    is_active_task = action in ["analyse_swe1", "analyse_swe2", "correct_swe1", "correct_swe2"] and active_status in ["running", "paused"]
    is_trace_active = action == "compare_trace"
    
    if (has_swe1 or has_swe2) and action and (is_active_task or is_trace_active):
        st.markdown("---")
        st.subheader("📊 Execution Results")
        
        action = st.session_state.last_action
        swe1_reqs = load_uploaded_requirements(swe1_files) if has_swe1 else []
        swe2_reqs = load_uploaded_requirements(swe2_files) if has_swe2 else []
        df = pd.DataFrame()
        
        def color_trace(val):
            color = '#d4edda' if val == 'Covered' else '#f8d7da'
            text_color = '#155724' if val == 'Covered' else '#721c24'
            return f'background-color: {color}; color: {text_color}; font-weight: bold;'
            
        if action == "analyse_swe1" and active_status in ["running", "paused"]:
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
                if is_running:
                    should_rerun = True
                
        elif action == "analyse_swe2" and active_status in ["running", "paused"]:
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
                if is_running:
                    should_rerun = True
                
        elif action == "correct_swe1" and active_status in ["running", "paused"]:
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
                if is_running:
                    should_rerun = True
                    
        elif action == "correct_swe2" and active_status in ["running", "paused"]:
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
                if is_running:
                    should_rerun = True
                    
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
                


        # --- EXPORT REPORT DELIVERABLES ---
        if not df.empty:
            st.markdown("---")
            st.markdown("### 📥 Export Audit Deliverables")
            
            # Convert findings dataframe to CSV
            csv_buffer = df.to_csv(index=False).encode('utf-8')
            
            # Determine the active spec and metadata for the current action
            if action in ["analyse_swe1", "correct_swe1"]:
                active_reqs = swe1_reqs
                active_files = swe1_files
                action_suffix = "swe1"
            elif action in ["analyse_swe2", "correct_swe2"]:
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

    # --- WORKFLOW EXECUTION HISTORY PANEL ---
    if "requirement_history" in st.session_state and st.session_state.requirement_history:
        st.markdown("---")
        st.subheader("📜 Workflow Execution History")
        
        # Display clear history button
        col_clear_1, col_clear_2 = st.columns([1.5, 5])
        with col_clear_1:
            if st.button("🧹 Clear History", key="clear_history_btn", use_container_width=True):
                st.session_state.requirement_history = []
                for task_action in ["analyse_swe1", "analyse_swe2", "correct_swe1", "correct_swe2"]:
                    st.session_state.pop(f"{task_action}_added_to_history", None)
                st.success("History cleared successfully!")
                st.rerun()
                
        # Render expanders
        for idx, item in enumerate(st.session_state.requirement_history):
            is_expanded = (idx == 0)
            item_type = item["type"]
            emoji = "📊" if item_type == "Analyze" else "🛠️"
            item_number = len(st.session_state.requirement_history) - idx
            label = " (Latest)" if idx == 0 else ""
            title = f"{emoji} {item_type} Requirement #{item_number}{label} - {item['source_file']} ({item['timestamp']})"
            
            with st.expander(title, expanded=is_expanded):
                st.markdown(f"**📅 Timestamp:** `{item['timestamp']}` | **📂 Source:** `{item['source_file']}`")
                
                df_item = pd.DataFrame(item["result"])
                
                if item_type == "Analyze":
                    total = len(df_item)
                    passed = sum(1 for r in item["result"] if r.get("Status") == "Passed")
                    review = total - passed
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Requirements Checked", total)
                    m2.metric("Passed", passed)
                    m3.metric("Review Needed", review)
                    
                    st.dataframe(apply_df_styling(df_item.style, color_status, subset=['Status']), use_container_width=True)
                else:
                    total = len(df_item)
                    changes_made = sum(1 for r in item["result"] if r.get("Original Requirement") != r.get("Corrected Requirement"))
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Requirements Checked", total)
                    m2.metric("Corrections Made", changes_made)
                    
                    st.dataframe(df_item, use_container_width=True)
                
                csv_buffer = df_item.to_csv(index=False).encode('utf-8')
                
                if item_type == "Analyze":
                    md_report = st.session_state.analyzer.generate_report(item["result"], [], item["source_file"])
                else:
                    md_report = st.session_state.analyzer.generate_report([], item["result"], item["source_file"])
                    
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button(
                        label="📥 Download Detailed Findings (.CSV)",
                        data=csv_buffer,
                        file_name=f"{item_type.lower()}_findings_{item['id'][:8]}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{item['id']}",
                        use_container_width=True
                    )
                with dcol2:
                    st.download_button(
                        label="📥 Download Compliance Report (.MD)",
                        data=md_report.encode('utf-8'),
                        file_name=f"{item_type.lower()}_compliance_report_{item['id'][:8]}.md",
                        mime="text/markdown",
                        key=f"dl_md_{item['id']}",
                        use_container_width=True
                    )

    if should_rerun:
        st.rerun()
