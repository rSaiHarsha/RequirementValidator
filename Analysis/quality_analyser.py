import re
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from Model.requirement import Requirement

def split_ears(text: str) -> (str, str):
    """
    Splits an EARS requirement into (precondition_prefix, action_part).
    The action part typically starts with "the [System Name] shall/should/must/will".
    """
    # Search for the action subject followed by a modal verb (shall, should, must, will)
    match = re.search(r"(?i)\bthe\s+\[?[^\]\n]+\]?\s+(?:shall|should|must|will)\b", text)
    if match:
        split_idx = match.start()
        return text[:split_idx], text[split_idx:]
    
    # Fallback to splitting at "the [" if no modal verb is found
    match_fallback = re.search(r"(?i)\bthe\s+\[", text)
    if match_fallback:
        split_idx = match_fallback.start()
        return text[:split_idx], text[split_idx:]
        
    return "", text

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

def analyze_single_requirement(index, r, llm, rag, rag_context=None):
    try:
        prefix, action_part = split_ears(r.content)
        
        if rag_context is None:
            rag_context = ""
            if rag:
                try:
                    rag_context = rag.query(action_part, top_k=2)
                except Exception:
                    pass
                
        system_prompt = (
            "You are an expert systems engineering auditor specializing in ASPICE and INCOSE requirement standards.\n"
            "Your task is to analyze the EARS system response (action part) of a requirement.\n"
            "Under INCOSE/ASPICE rules:\n"
            "1. The statement MUST use the standard modal verb 'shall' (do not use should, will, must, or behaves).\n"
            "2. The statement MUST NOT contain vague or subjective terms (e.g. fast, quickly, beautifully, great, modern, creepy).\n"
            "3. The statement MUST be verifiable and measurable.\n"
            "4. The statement must not combine multiple distinct requirements (e.g. multiple actions).\n"
            "\nYou are given both the Full Requirement (for complete context such as preconditions and triggers) and the specific Action Part under audit. Focus your compliance assessment and quality check on the Action Part using the Full Requirement as context."
        )
        if rag_context:
            system_prompt += (
                "\nIn addition to the standard rules, you MUST also enforce these project-specific rules retrieved from the knowledge base:\n"
                f"{rag_context}\n"
            )
        system_prompt += (
            "\nAnalyze the action part. If it violates any rules, return 'Review', name the broken rule, and explain why.\n"
            "Otherwise, return 'Passed'.\n\n"
            "You must return your output strictly in JSON format matching this schema:\n"
            "{\n"
            "  \"status\": \"Passed\" or \"Review\",\n"
            "  \"failed_rule\": \"Rule name\" or \"None\",\n"
            "  \"rationale\": \"Detailed explanation of the audit decision\"\n"
            "}\n"
            "Do not include any explanation or markdown formatting outside the JSON."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Full Requirement: \"{r.content}\"\nAction Part under audit: \"{action_part}\"\nOriginal Rationale: \"{r.rationale}\""}
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

def analyze_batch(batch_items, llm, rag):
    # batch_items is a list of tuples: (index, Requirement)
    # Returns a list of dicts mapping index -> analysis_result
    
    # 1. Split EARS and retrieve RAG rules in batch
    req_details = []
    action_parts = []
    for idx, r in batch_items:
        prefix, action_part = split_ears(r.content)
        req_details.append({
            "index": idx,
            "r": r,
            "prefix": prefix,
            "action_part": action_part,
            "rag_context": ""
        })
        action_parts.append(action_part)
        
    if rag:
        try:
            rag_contexts = rag.query_batch(action_parts, top_k=2)
            for i, ctx in enumerate(rag_contexts):
                req_details[i]["rag_context"] = ctx
        except Exception:
            pass
    
    # 2. Build prompts
    system_prompt = (
        "You are an expert systems engineering auditor specializing in ASPICE and INCOSE requirement standards.\n"
        "Your task is to analyze a batch of EARS system responses (action parts) of requirements.\n"
        "Under INCOSE/ASPICE rules:\n"
        "1. The statement MUST use the standard modal verb 'shall' (do not use should, will, must, or behaves).\n"
        "2. The statement MUST NOT contain vague or subjective terms (e.g. fast, quickly, beautifully, great, modern, creepy).\n"
        "3. The statement MUST be verifiable and measurable.\n"
        "4. The statement must not combine multiple distinct requirements (e.g. multiple actions).\n"
        "For each requirement in the batch, check these rules and any project-specific rules provided.\n\n"
        "\nYou are given both the Full Requirement (for complete context such as preconditions and triggers) and the specific Action Part under audit. Focus your compliance assessment and quality check on the Action Part using the Full Requirement as context."
        "You must return your output strictly in JSON format matching this schema:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"index\": 0,\n"
        "      \"id\": \"Requirement ID (string)\",\n"
        "      \"status\": \"Passed\" or \"Review\",\n"
        "      \"failed_rule\": \"Rule name\" or \"None\",\n"
        "      \"rationale\": \"Detailed explanation of the audit decision\"\n"
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
            f"Action Statement: \"{item['action_part']}\"\n"
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

def analyze_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None) -> List[Dict[str, Any]]:
    total = len(requirements)
    analysis_data = [None] * total
    
    batch_size = 5
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return analyze_batch(batch, llm, rag)
        except Exception:
            # Fallback to single requirement analysis
            fallback_results = {}
            for idx, r in batch:
                _, res = analyze_single_requirement(idx, r, llm, rag)
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

def analyze_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single") -> List[Dict[str, Any]]:
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
        return analyze_requirements_batch(requirements, llm, progress_callback, rag)

    # Pre-compute RAG contexts in a single batch embeddings query
    rag_contexts = [""] * total
    if rag:
        try:
            action_parts = [split_ears(r.content)[1] for r in requirements]
            rag_contexts = rag.query_batch(action_parts, top_k=2)
        except Exception:
            pass

    analysis_data = [None] * total
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_single_requirement, i, r, llm, rag, rag_contexts[i]): i 
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

def correct_single_requirement(index, r, llm, rag, rag_context=None):
    try:
        prefix, action_part = split_ears(r.content)
        
        if rag_context is None:
            rag_context = ""
            if rag:
                try:
                    rag_context = rag.query(action_part, top_k=2)
                except Exception:
                    pass
    
        system_prompt = (
            "You are a systems engineering expert specializing in ASPICE, ISO 26262, and EARS.\n"
            "Your task is to correct/rewrite only the system response (action part) of a requirement if it violates INCOSE/ASPICE rules.\n"
            "Rules:\n"
            "1. Enforce the standard modal verb 'shall'. Replace should, will, must, behaves.\n"
            "2. Remove vague or subjective terms and replace them with specific, measurable criteria.\n"
        )
        if rag_context:
            system_prompt += (
                "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
                f"{rag_context}\n"
            )
        system_prompt += (
            "\n3. If the action part is already compliant and uses 'shall' properly without vague terms, return it EXACTLY as-is.\n"
            "4. Return ONLY the corrected action part. Do not include prefix clauses, trigger phrases (like When, If), explanations, quotes, or markdown."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Action Part: \"{action_part}\"\nIssue: \"{r.rationale}\""}
        ]
        
        response = llm.get_response(messages, stream=False)
        res_text = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if res_text.startswith("```"):
            lines = res_text.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].endswith("```"):
                    lines = lines[:-1]
                res_text = "\n".join(lines).strip()
        # Clean quotes if any
        if (res_text.startswith('"') and res_text.endswith('"')) or (res_text.startswith("'") and res_text.endswith("'")):
            res_text = res_text[1:-1].strip()
            
        corrected_action = res_text if res_text else action_part
        full_corrected = prefix + corrected_action
        
        return index, r, action_part, corrected_action, full_corrected
    except Exception as e:
        return index, r, action_part, f"LLM Error: {str(e)}", prefix + action_part

def correct_batch(batch_items, llm, rag):
    # batch_items: list of (idx, r)
    req_details = []
    action_parts = []
    for idx, r in batch_items:
        prefix, action_part = split_ears(r.content)
        req_details.append({
            "index": idx,
            "r": r,
            "prefix": prefix,
            "action_part": action_part,
            "rag_context": ""
        })
        action_parts.append(action_part)
        
    if rag:
        try:
            rag_contexts = rag.query_batch(action_parts, top_k=2)
            for i, ctx in enumerate(rag_contexts):
                req_details[i]["rag_context"] = ctx
        except Exception:
            pass
        
    system_prompt = (
        "You are a systems engineering expert specializing in ASPICE, ISO 26262, and EARS.\n"
        "Your task is to analyze a batch of requirements and correct/rewrite only the system response (action part) if it violates INCOSE/ASPICE rules.\n"
        "Rules:\n"
        "1. Enforce the standard modal verb 'shall'. Replace should, will, must, behaves.\n"
        "2. Remove vague or subjective terms and replace them with specific, measurable criteria.\n"
        "3. If the action part is already compliant and uses 'shall' properly without vague terms, keep it EXACTLY as-is.\n\n"
        "For each requirement in the batch, analyze and correct only the action statement part.\n\n"
        "You must return your output strictly in JSON format matching this schema:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"index\": 0,\n"
        "      \"id\": \"Requirement ID (string)\",\n"
        "      \"corrected_action\": \"The corrected action part only, preserving standard formatting. If already compliant, return the original action part verbatim.\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Do not include any prefix clauses, trigger phrases (like When, If), explanations, quotes, or markdown outside the JSON."
    )
    
    user_content = "Please correct/rewrite the action statement of the following batch of requirements:\n\n"
    for item in req_details:
        user_content += (
            f"Index: {item['index']}\n"
            f"ID: {item['r'].name}\n"
            f"Action Part: \"{item['action_part']}\"\n"
            f"Issue: \"{item['r'].rationale}\"\n"
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
            corrected_action = res.get("corrected_action", "")
            if corrected_action is None:
                corrected_action = ""
            else:
                corrected_action = str(corrected_action).strip()
            # Clean markdown code blocks and quotes
            if corrected_action.startswith("```"):
                lines = corrected_action.splitlines()
                if len(lines) >= 2:
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].endswith("```"):
                        lines = lines[:-1]
                    corrected_action = "\n".join(lines).strip()
            if (corrected_action.startswith('"') and corrected_action.endswith('"')) or (corrected_action.startswith("'") and corrected_action.endswith("'")):
                corrected_action = corrected_action[1:-1].strip()
                
            if not corrected_action:
                corrected_action = item['action_part']
                
            full_corrected = item['prefix'] + corrected_action
            batch_results[idx] = (item['r'], item['action_part'], corrected_action, full_corrected)
        else:
            raise ValueError(f"Requirement index {idx} / ID '{item['r'].name}' not found in LLM response.")
            
    return batch_results

def correct_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None) -> List[Dict[str, Any]]:
    total = len(requirements)
    correction_data_map = {}
    
    batch_size = 5
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return correct_batch(batch, llm, rag)
        except Exception:
            # Fallback to single requirement correction
            fallback_results = {}
            for idx, r in batch:
                _, r_obj, action_part, corrected_action, full_corrected = correct_single_requirement(idx, r, llm, rag)
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

def correct_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single") -> List[Dict[str, Any]]:
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
        return correct_requirements_batch(requirements, llm, progress_callback, rag)

    # Pre-compute RAG contexts in a single batch embeddings query
    rag_contexts = [""] * total
    if rag:
        try:
            action_parts = [split_ears(r.content)[1] for r in requirements]
            rag_contexts = rag.query_batch(action_parts, top_k=2)
        except Exception:
            pass

    correction_data_map = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(correct_single_requirement, i, r, llm, rag, rag_contexts[i]): i 
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
