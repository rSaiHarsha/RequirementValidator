import re
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from Model.requirement import Requirement


def clean_and_parse_json(text: str) -> dict:
    """Helper to safely extract and parse a JSON block from LLM markdown response."""
    if not text or not isinstance(text, str):
        raise ValueError("LLM response is empty or not a string.")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON block found in LLM response.")
    text = text[start:end+1]
    return json.loads(text)

def analyze_single_requirement(index, r, llm, rag, rag_context=None, selected_collections=None):
    try:
        if rag_context is None:
            rag_context = ""
            if rag:
                try:
                    rag_context = rag.query(r.content, collection_name=selected_collections, top_k=2)
                except Exception:
                    pass
                
        system_prompt = (
            "You are an expert systems engineering auditor specializing in ASPICE, INCOSE, and EARS requirement standards.\n"
            "Your task is to structurally parse and analyze an engineering requirement.\n"
            "1. EARS Syntax (Preconditions/Triggers): If the requirement has preconditions or triggers, they must start with EARS keywords (If, When, While, Where).\n"
            "   (Note: If a precondition violates EARS syntax but the action part is flawless, flag it as a warning in your rationale but keep the Status as 'Passed'. If it's severely malformed or confusing, flag 'Review').\n"
            "2. INCOSE Rules (System Response / Action Part):\n"
            "   - MUST use the standard modal verb 'shall' (do not use should, will, must, or behaves).\n"
            "   - MUST NOT contain vague or subjective terms (e.g. fast, quickly, beautifully, great, modern, creepy).\n"
            "   - MUST be verifiable and measurable.\n"
            "   - MUST NOT combine multiple distinct requirements (e.g. multiple actions).\n"
        )
        if rag_context:
            system_prompt += (
                "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
                f"{rag_context}\n"
            )
        system_prompt += (
            "\nAnalyze the requirement structurally. Parse it internally into Preconditions, System Name, Modality, and System Response. Then evaluate the rules.\n"
            "If it violates critical INCOSE rules, return 'Review', name the broken rule, and explain why.\n"
            "Otherwise, return 'Passed'.\n\n"
            "You must return your output strictly in JSON format matching this schema:\n"
            "{\n"
            "  \"status\": \"Passed\" or \"Review\",\n"
            "  \"failed_rule\": \"Rule name\" or \"None\",\n"
            "  \"rationale\": \"Detailed explanation of the structural parsing and audit decision\"\n"
            "}\n"
            "Do not include any explanation or markdown formatting outside the JSON."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Full Requirement: \"{r.content}\"\nOriginal Rationale: \"{r.rationale}\""}
        ]
        
        response = llm.get_response(messages, stream=False)
        data = clean_and_parse_json(response.choices[0].message.content)
        
        return index, {
            "ID": r.name,
            "Requirement": r.content,
            "State": r.state,
            "ASIL": r.asil,
            "Status": data.get("status", "Passed"),
            "Failed Rule": data.get("failed_rule", "None"),
            "Rationale": data.get("rationale", "Complies with EARS/INCOSE rules")
        }
    except Exception as e:
        return index, {
            "ID": r.name,
            "Requirement": r.content,
            "State": r.state,
            "ASIL": r.asil,
            "Status": "Review",
            "Failed Rule": "LLM Error",
            "Rationale": f"LLM analysis failed: {str(e)}"
        }

def analyze_batch(batch_items, llm, rag, selected_collections=None):
    # batch_items is a list of tuples: (index, Requirement)
    # Returns a list of dicts mapping index -> analysis_result
    
    req_details = []
    full_reqs = []
    for idx, r in batch_items:
        req_details.append({
            "index": idx,
            "r": r,
            "rag_context": ""
        })
        full_reqs.append(r.content)
        
    if rag:
        try:
            rag_contexts = rag.query_batch(full_reqs, collection_name=selected_collections, top_k=2)
            for i, ctx in enumerate(rag_contexts):
                req_details[i]["rag_context"] = ctx
        except Exception:
            pass
    
    # 2. Build prompts
    system_prompt = (
        "You are an expert systems engineering auditor specializing in ASPICE, INCOSE, and EARS requirement standards.\n"
        "Your task is to structurally parse and analyze a batch of engineering requirements.\n"
        "1. EARS Syntax (Preconditions/Triggers): If the requirement has preconditions or triggers, they must start with EARS keywords (If, When, While, Where).\n"
        "   (Note: If the precondition violates EARS syntax but the action part is flawless, flag it as a warning in your rationale but keep the Status as 'Passed'. If severely malformed, flag 'Review').\n"
        "2. INCOSE Rules (System Response / Action Part):\n"
        "   - MUST use the standard modal verb 'shall' (do not use should, will, must, or behaves).\n"
        "   - MUST NOT contain vague or subjective terms (e.g. fast, quickly, beautifully, great, modern, creepy).\n"
        "   - MUST be verifiable and measurable.\n"
        "   - MUST NOT combine multiple distinct requirements (e.g. multiple actions).\n"
        "For each requirement in the batch, check these rules and any project-specific rules provided.\n\n"
        "\nStructurally parse each requirement internally into Preconditions, System Name, Modality, and System Response. Then evaluate the rules.\n"
        "You must return your output strictly in JSON format matching this schema:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"index\": 0,\n"
        "      \"id\": \"Requirement ID (string)\",\n"
        "      \"status\": \"Passed\" or \"Review\",\n"
        "      \"failed_rule\": \"Rule name\" or \"None\",\n"
        "      \"rationale\": \"Detailed explanation of the structural parsing and audit decision\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Do not include any explanation or markdown formatting outside the JSON."
    )
    
    user_content = "Please analyze the following batch of requirements:\n\n"
    for item in req_details:
        user_content += (
            f"Index: {item['index']}\n"
            f"ID: {item['r'].name}\n"
            f"Full Requirement: \"{item['r'].content}\"\n"
            f"Original Rationale: \"{item['r'].rationale}\"\n"
        )
        if item['rag_context']:
            user_content += f"Project-specific rules: \"{item['rag_context']}\"\n"
        user_content += "---\n"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    response = llm.get_response(messages, stream=False)
    raw_response = response.choices[0].message.content
    data = clean_and_parse_json(raw_response)
    
    results_by_index = {}
    for res in data.get("results", []):
        if "index" in res:
            try:
                results_by_index[int(res["index"])] = res
            except Exception:
                pass
                
    batch_results = {}
    for item in req_details:
        idx = item['index']
        res = None
        if idx in results_by_index:
            res = results_by_index[idx]
        else:
            # Try to lookup by ID
            r_id = str(item['r'].name).strip().lower()
            for r_res in data.get("results", []):
                if str(r_res.get("id")).strip().lower() == r_id:
                    res = r_res
                    break
        
        if res is not None:
            batch_results[idx] = {
                "ID": item['r'].name,
                "Requirement": item['r'].content,
                "State": item['r'].state,
                "ASIL": item['r'].asil,
                "Status": res.get("status", "Passed"),
                "Failed Rule": res.get("failed_rule", "None"),
                "Rationale": res.get("rationale", "Complies with EARS/INCOSE rules")
            }
        else:
            raise ValueError(f"Requirement index {idx} / ID '{item['r'].name}' not found in LLM response.")
            
    return batch_results

def analyze_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None, selected_collections=None) -> List[Dict[str, Any]]:
    total = len(requirements)
    analysis_data = [None] * total
    
    batch_size = 5
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return analyze_batch(batch, llm, rag, selected_collections)
        except Exception:
            # Fallback to single requirement analysis
            fallback_results = {}
            for idx, r in batch:
                _, res = analyze_single_requirement(idx, r, llm, rag, selected_collections=selected_collections)
                fallback_results[idx] = res
            return fallback_results

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        
        completed_count = 0
        for future in as_completed(futures):
            batch_res = future.result()
            for idx, res in batch_res.items():
                analysis_data[idx] = res
                completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total)
                
    return analysis_data

def analyze_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None) -> List[Dict[str, Any]]:
    """
    Perform an AI-driven ASPICE/INCOSE audit using NVIDIA LLM concurrently or in batches,
    concentrating on the EARS action statement and referencing RAG rules if available.
    """
    if not llm:
        raise ValueError("LLMManager is required for quality analysis.")

    total = len(requirements)
    if total == 0:
        return []

    if mode == "batch":
        return analyze_requirements_batch(requirements, llm, progress_callback, rag, selected_collections)

    # Pre-compute RAG contexts in a single batch embeddings query
    rag_contexts = [""] * total
    if rag:
        try:
            action_parts = [split_ears(r.content)[1] for r in requirements]
            rag_contexts = rag.query_batch(action_parts, collection_name=selected_collections, top_k=2)
        except Exception:
            pass

    analysis_data = [None] * total
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_single_requirement, i, r, llm, rag, rag_contexts[i], selected_collections): i 
            for i, r in enumerate(requirements)
        }
        
        completed_count = 0
        for future in as_completed(futures):
            index, result = future.result()
            analysis_data[index] = result
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total)
                
    return analysis_data

def correct_single_requirement(index, r, llm, rag, rag_context=None, selected_collections=None):
    try:
        if rag_context is None:
            rag_context = ""
            if rag:
                try:
                    rag_context = rag.query(r.content, collection_name=selected_collections, top_k=2)
                except Exception:
                    pass
    
        system_prompt = (
            "You are a systems engineering expert specializing in ASPICE, ISO 26262, and EARS.\n"
            "Your task is to correct/rewrite a flawed engineering requirement by structurally parsing it and fixing only the sections that violate rules.\n"
            "Rules:\n"
            "1. EARS Syntax: Ensure preconditions/triggers start with standard EARS keywords (If, When, While, Where). Rewrite them if they are malformed.\n"
            "2. Modality: Enforce the standard modal verb 'shall'. Replace should, will, must, behaves.\n"
            "3. Action Part: Remove vague or subjective terms and replace them with specific, measurable criteria.\n"
        )
        if rag_context:
            system_prompt += (
                "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
                f"{rag_context}\n"
            )
        system_prompt += (
            "\n4. If the requirement is already fully compliant, return it EXACTLY as-is.\n"
            "5. Reconstruct the fully corrected requirement string. Return ONLY the fully corrected requirement string. Do not include explanations, quotes, or markdown format blocks."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Full Requirement Context: \"{r.content}\""}
        ]
        
        response = llm.get_response(messages, stream=False)
        full_corrected = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks
        if full_corrected.startswith("```"):
            lines = full_corrected.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].endswith("```"):
                    lines = lines[:-1]
                full_corrected = "\n".join(lines).strip()
        # Clean quotes if any
        if (full_corrected.startswith('"') and full_corrected.endswith('"')) or (full_corrected.startswith("'") and full_corrected.endswith("'")):
            full_corrected = full_corrected[1:-1].strip()
            
        if not full_corrected:
            full_corrected = r.content
            
        return index, r, r.content, full_corrected, full_corrected
    except Exception as e:
        return index, r, r.content, f"LLM Error: {str(e)}", r.content

def correct_batch(batch_items, llm, rag, selected_collections=None):
    # batch_items: list of (idx, r)
    req_details = []
    full_reqs = []
    for idx, r in batch_items:
        req_details.append({
            "index": idx,
            "r": r,
            "rag_context": ""
        })
        full_reqs.append(r.content)
        
    if rag:
        try:
            rag_contexts = rag.query_batch(full_reqs, collection_name=selected_collections, top_k=2)
            for i, ctx in enumerate(rag_contexts):
                req_details[i]["rag_context"] = ctx
        except Exception:
            pass
        
    system_prompt = (
        "You are a systems engineering expert specializing in ASPICE, ISO 26262, and EARS.\n"
        "Your task is to analyze a batch of requirements and correct/rewrite them structurally.\n"
        "Rules:\n"
        "1. EARS Syntax: Ensure preconditions/triggers start with standard EARS keywords (If, When, While, Where). Rewrite them if they are malformed.\n"
        "2. Modality: Enforce the standard modal verb 'shall'. Replace should, will, must, behaves.\n"
        "3. Action Part: Remove vague or subjective terms and replace them with specific, measurable criteria.\n"
        "4. If the requirement is already compliant, keep it EXACTLY as-is.\n\n"
        "For each requirement in the batch, structurally parse and rewrite it if needed. Reconstruct the fully corrected requirement.\n\n"
        "You must return your output strictly in JSON format matching this schema:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"index\": 0,\n"
        "      \"id\": \"Requirement ID (string)\",\n"
        "      \"full_corrected\": \"The fully corrected requirement string.\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Do not include any explanations, quotes, or markdown outside the JSON."
    )
    
    user_content = "Please correct/rewrite the following batch of requirements:\n\n"
    for item in req_details:
        user_content += (
            f"Index: {item['index']}\n"
            f"ID: {item['r'].name}\n"
            f"Full Requirement Context: \"{item['r'].content}\"\n"
        )
        if item['rag_context']:
            user_content += f"Project-specific rules: \"{item['rag_context']}\"\n"
        user_content += "---\n"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    response = llm.get_response(messages, stream=False)
    raw_response = response.choices[0].message.content
    data = clean_and_parse_json(raw_response)
    
    results_by_index = {}
    for res in data.get("results", []):
        if "index" in res:
            try:
                results_by_index[int(res["index"])] = res
            except Exception:
                pass
                
    batch_results = {}
    for item in req_details:
        idx = item['index']
        res = None
        if idx in results_by_index:
            res = results_by_index[idx]
        else:
            r_id = str(item['r'].name).strip().lower()
            for r_res in data.get("results", []):
                if str(r_res.get("id")).strip().lower() == r_id:
                    res = r_res
                    break
                    
        if res is not None:
            full_corrected = res.get("full_corrected", "")
            if full_corrected is None:
                full_corrected = ""
            else:
                full_corrected = str(full_corrected).strip()
                
            # Clean markdown code blocks and quotes
            if full_corrected.startswith("```"):
                lines = full_corrected.splitlines()
                if len(lines) >= 2:
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].endswith("```"):
                        lines = lines[:-1]
                    full_corrected = "\n".join(lines).strip()
            if (full_corrected.startswith('"') and full_corrected.endswith('"')) or (full_corrected.startswith("'") and full_corrected.endswith("'")):
                full_corrected = full_corrected[1:-1].strip()
                
            if not full_corrected:
                full_corrected = item['r'].content
                
            batch_results[idx] = (item['r'], item['r'].content, full_corrected, full_corrected)
        else:
            raise ValueError(f"Requirement index {idx} / ID '{item['r'].name}' not found in LLM response.")
            
    return batch_results

def correct_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None, selected_collections=None) -> List[Dict[str, Any]]:
    total = len(requirements)
    correction_data_map = {}
    
    batch_size = 5
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return correct_batch(batch, llm, rag, selected_collections)
        except Exception:
            # Fallback to single requirement correction
            fallback_results = {}
            for idx, r in batch:
                _, r_obj, action_part, corrected_action, full_corrected = correct_single_requirement(idx, r, llm, rag, selected_collections=selected_collections)
                fallback_results[idx] = (r_obj, action_part, corrected_action, full_corrected)
            return fallback_results

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        
        completed_count = 0
        for future in as_completed(futures):
            batch_res = future.result()
            for idx, (r, action_part, corrected_action, full_corrected) in batch_res.items():
                completed_count += 1
                if corrected_action.strip().lower() != action_part.strip().lower():
                    correction_data_map[idx] = {
                        "ID": r.name,
                        "Original Requirement": r.content,
                        "Rationale of Issue": r.rationale if r.rationale else "Identified non-compliance in EARS action part",
                        "Corrected Requirement": full_corrected
                    }
            if progress_callback:
                progress_callback(completed_count, total)
                
    correction_data = [correction_data_map[k] for k in sorted(correction_data_map.keys())]
    return correction_data

def correct_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None) -> List[Dict[str, Any]]:
    """
    Rewrite problematic requirements using the LLM concurrently or in batches, preserving precondition prefixes
    and applying RAG baseline knowledge rules if uploaded.
    """
    if not llm:
        raise ValueError("LLMManager is required for requirement corrections.")

    total = len(requirements)
    if total == 0:
        return []

    if mode == "batch":
        return correct_requirements_batch(requirements, llm, progress_callback, rag, selected_collections)

    # Pre-compute RAG contexts in a single batch embeddings query
    rag_contexts = [""] * total
    if rag:
        try:
            action_parts = [split_ears(r.content)[1] for r in requirements]
            rag_contexts = rag.query_batch(action_parts, collection_name=selected_collections, top_k=2)
        except Exception:
            pass

    correction_data_map = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(correct_single_requirement, i, r, llm, rag, rag_contexts[i], selected_collections): i 
            for i, r in enumerate(requirements)
        }
        
        completed_count = 0
        for future in as_completed(futures):
            index, r, action_part, corrected_action, full_corrected = future.result()
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total)
                
            if corrected_action.strip().lower() != action_part.strip().lower():
                correction_data_map[index] = {
                    "ID": r.name,
                    "Original Requirement": r.content,
                    "Rationale of Issue": r.rationale if r.rationale else "Identified non-compliance in EARS action part",
                    "Corrected Requirement": full_corrected
                }
                
    # Reassemble correction items ordered by original sequence
    correction_data = [correction_data_map[k] for k in sorted(correction_data_map.keys())]
    return correction_data

def generate_markdown_report(analysis_results: List[Dict[str, Any]], correction_results: List[Dict[str, Any]], file_title: str) -> str:
    """
    Generates a professional Markdown quality report for download.
    """
    total = len(analysis_results)
    passed = sum(1 for item in analysis_results if item["Status"] == "Passed")
    review = total - passed
    score = int((passed / total) * 100) if total > 0 else 0
    
    md = [
        f"# EARS/INCOSE Requirements Quality Audit Report",
        f"**Target Specification File**: `{file_title}`",
        f"**Overall Compliance Score**: `{score}%`",
        f"",
        f"## 1. Executive Summary",
        f"- **Total Checked Requirements**: {total}",
        f"- **Passed (Compliant)**: {passed}",
        f"- **Review Needed (Non-compliant)**: {review}",
        f"",
        f"## 2. Detailed Quality Audit Findings",
        f"| Requirement ID | Audit Status | Failed Rule | Rationale |",
        f"| :--- | :--- | :--- | :--- |"
    ]
    
    for item in analysis_results:
        clean_rat = item["Rationale"].replace("\n", " ").replace("|", "\\|")
        md.append(f"| {item['ID']} | {item['Status']} | {item['Failed Rule']} | {clean_rat} |")
        
    if correction_results:
        md.extend([
            f"",
            f"## 3. Recommended EARS Action-Part Rewrites",
            f"| ID | Original Requirement | Corrected Requirement (Preserved Precondition) |",
            f"| :--- | :--- | :--- |"
        ])
        for corr in correction_results:
            clean_orig = corr["Original Requirement"].replace("\n", " ").replace("|", "\\|")
            clean_corr = corr["Corrected Requirement"].replace("\n", " ").replace("|", "\\|")
            md.append(f"| {corr['ID']} | {clean_orig} | {clean_corr} |")
            
    return "\n".join(md)
