import re
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from Model.requirement import Requirement


def get_effective_system_prompt(default_prompt: str, mode: str = "analysis") -> str:
    """Helper function to apply Prompt Sandbox override if enabled."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            if st.session_state.get(f"use_custom_prompt_{mode}", False):
                custom_prompt = st.session_state.get(f"custom_prompt_{mode}", "").strip()
                if custom_prompt:
                    return custom_prompt
    except Exception:
        pass
    return default_prompt

def clean_and_parse_json(text: str):
    """Helper to safely extract and parse a JSON block from LLM markdown response."""
    if not text or not isinstance(text, str):
        raise ValueError("LLM response is empty or not a string.")
        
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    
    # Determine if it's an array or object based on which brackets enclose the content
    is_array = False
    if start_arr != -1 and end_arr != -1:
        if start_obj == -1 or start_arr < start_obj:
            is_array = True
            
    if is_array:
        start = start_arr
        end = end_arr
    else:
        start = start_obj
        end = end_obj
        
    if start == -1 or end == -1:
        raise ValueError("No JSON block found in LLM response.")
    text = text[start:end+1]
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*\]', ']', text)
        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON even after cleanup: {str(e)}")
            
    # Normalize dictionary to guarantee deterministic key/values
    if isinstance(data, dict):
        normalized_data = {}
        for k, v in data.items():
            key = k.strip()
            if isinstance(v, str):
                v = v.strip()
            if key.lower() == "status" and isinstance(v, str):
                v = "Passed" if v.lower() == "passed" else "Review"
            normalized_data[key.lower()] = v
        return normalized_data
    elif isinstance(data, list):
        normalized_list = []
        for item in data:
            if isinstance(item, dict):
                normalized_data = {}
                for k, v in item.items():
                    key = k.strip()
                    if isinstance(v, str):
                        v = v.strip()
                    if key.lower() == "status" and isinstance(v, str):
                        v = "Passed" if v.lower() == "passed" else "Review"
                    normalized_data[key.lower()] = v
                normalized_list.append(normalized_data)
            else:
                normalized_list.append(item)
        return normalized_list
    return data

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
            "You are a strict, deterministic Systems Engineering Requirements Auditor.\n"
            "Your task is to analyze an engineering requirement using INCOSE guidelines and EARS syntax.\n"
            "You MUST:\n"
            "- Identify structural components (trigger, condition, system response)\n"
            "- Evaluate compliance against INCOSE guidelines, EARS syntax\n"
            "\n"
            "Rules for Output:\n"
            "1. Return ONLY valid JSON exactly matching the schema below.\n"
            "2. Do NOT include any explanation or markdown formatting outside the JSON.\n"
            "3. Do NOT invent information. Output must be perfectly reproducible.\n"
            "\n"
            "JSON Schema:\n"
            "{\n"
            "  \"status\": \"Passed\" or \"Review\",\n"
            "  \"failed_rule\": \"Rule name\" or \"None\",\n"
            "  \"rationale\": \"Concise structured explanation\"\n"
            "}"
        )
        
            
                
            
        system_prompt += (
            "\nAnalyze the requirement structurally. Parse it internally into Preconditions, System Name, Modality, and System Response. Then evaluate the rules.\n"
            "If it violates critical INCOSE rules and EARS Syntax, status MUST be 'Review'. Name the broken rule, and explain why.\n"
            "Otherwise, status MUST be 'Passed'."
        )
        
        system_prompt = get_effective_system_prompt(system_prompt, mode="analysis")
        if rag_context:
            system_prompt += (
"\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
                f"{rag_context}\n"
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
    results = {}
    if not batch_items:
        return results

    rag_contexts = []
    if rag:
        try:
            reqs_text = [r.content for _, r in batch_items]
            rag_contexts = rag.query_batch(reqs_text, collection_name=selected_collections, top_k=2)
        except Exception:
            rag_contexts = [""] * len(batch_items)
    else:
        rag_contexts = [""] * len(batch_items)

    system_prompt = (
        "You are a strict, deterministic Systems Engineering Requirements Auditor.\n"
        "Your task is to analyze multiple engineering requirements using INCOSE guidelines and EARS syntax.\n"
        "You MUST evaluate each requirement structurally and check compliance.\n\n"
        "Rules for Output:\n"
        "1. Return ONLY valid JSON exactly matching the schema below.\n"
        "2. Do NOT include any explanation or markdown formatting outside the JSON.\n"
        "3. The output MUST be a JSON array of objects, in the EXACT same order as the inputs.\n\n"
        "JSON Schema:\n"
        "[\n"
        "  {\n"
        "    \"status\": \"Passed\" or \"Review\",\n"
        "    \"failed_rule\": \"Rule name\" or \"None\",\n"
        "    \"rationale\": \"Concise structured explanation\"\n"
        "  }\n"
        "]"
    )

    combined_rag_context = ""
    for i, ctx in enumerate(rag_contexts):
        if ctx:
            combined_rag_context += f"Context for Requirement {i+1}:\n{ctx}\n"

    

    system_prompt = get_effective_system_prompt(system_prompt, mode="batch_analysis")
    if combined_rag_context:
        system_prompt += (
            "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
            f"{combined_rag_context}\n"
        )    
    user_content = "Analyze the following requirements:\n\n"
    for i, (idx, r) in enumerate(batch_items):
        user_content += f"--- Requirement {i+1} ---\n"
        user_content += f"ID: {r.name}\n"
        user_content += f"Text: \"{r.content}\"\n"
        user_content += f"Rationale: \"{r.rationale}\"\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    try:
        response = llm.get_response(messages, stream=False)
        raw_text = response.choices[0].message.content
        data = clean_and_parse_json(raw_text)
        
        if not isinstance(data, list) or len(data) != len(batch_items):
            raise ValueError("LLM did not return an array of the correct length.")
            
        for i, (idx, r) in enumerate(batch_items):
            item_data = data[i]
            results[idx] = {
                "ID": r.name,
                "Requirement": r.content,
                "State": r.state,
                "ASIL": r.asil,
                "Status": item_data.get("status", "Passed"),
                "Failed Rule": item_data.get("failed_rule", "None"),
                "Rationale": item_data.get("rationale", "Complies with EARS/INCOSE rules")
            }
        return results
    except Exception as e:
        # Fallback to single parallel processing
        fallback_results = {}
        with ThreadPoolExecutor(max_workers=min(10, len(batch_items))) as ex:
            fs = {ex.submit(analyze_single_requirement, idx, r, llm, rag, rag_contexts[i] if rag_contexts else None, selected_collections): idx for i, (idx, r) in enumerate(batch_items)}
            for f in as_completed(fs):
                idx = fs[f]
                _, res = f.result()
                fallback_results[idx] = res
        return fallback_results

def analyze_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None, selected_collections=None, batch_size=10) -> List[Dict[str, Any]]:
    total = len(requirements)
    analysis_data = [None] * total
    
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return analyze_batch(batch, llm, rag, selected_collections)
        except Exception:
            fallback_results = {}
            with ThreadPoolExecutor(max_workers=min(10, len(batch))) as ex:
                fs = {ex.submit(analyze_single_requirement, idx, r, llm, rag, selected_collections=selected_collections): idx for idx, r in batch}
                for f in as_completed(fs):
                    idx = fs[f]
                    _, res = f.result()
                    fallback_results[idx] = res
            return fallback_results

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        
        completed_count = 0
        for future in as_completed(futures):
            batch_res = future.result()
            for idx, res in batch_res.items():
                analysis_data[idx] = res
                completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total, [x for x in analysis_data if x is not None])
                
    return analysis_data

def analyze_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None, batch_size=10) -> List[Dict[str, Any]]:
    if not llm:
        raise ValueError("LLMManager is required for quality analysis.")

    total = len(requirements)
    if total == 0:
        return []

    if mode == "batch":
        return analyze_requirements_batch(requirements, llm, progress_callback, rag, selected_collections, batch_size)

    rag_contexts = [""] * total
    if rag:
        try:
            full_reqs = [r.content for r in requirements]
            rag_contexts = rag.query_batch(full_reqs, collection_name=selected_collections, top_k=2)
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
                progress_callback(completed_count, total, [x for x in analysis_data if x is not None])
                
    return analysis_data

def correct_single_requirement(index, r, llm, rag, rag_context=None, selected_collections=None, feedback_rule=None, feedback_rationale=None, initial_text=None):
    max_retries = 3
    current_text = initial_text if initial_text is not None else r.content
    failed_rule = feedback_rule
    rationale = feedback_rationale
    
    try:
        if rag_context is None:
            rag_context = ""
            if rag:
                try:
                    rag_context = rag.query(r.content, collection_name=selected_collections, top_k=2)
                except Exception:
                    pass
    
        system_prompt = (
            "You are a strict, deterministic Senior Systems Engineer and Requirements Expert.\n"
            "Your task is to analyze and correct engineering requirements using INCOSE guidelines and EARS syntax.\n"
            "You MUST adhere to these strict rules:\n"
            "1. Return ONLY valid JSON exactly matching the schema below.\n"
            "2. Do NOT include any explanation or markdown formatting outside the JSON.\n"
            "3. Do NOT invent information. Output must be perfectly reproducible.\n"
            "4. Split the requirement if it contains multiple actions.\n"
            "\n"
            "JSON Schema:\n"
            "{\n"
            "  \"split_required\": boolean,\n"
            "  \"corrected_requirements\": [string]\n"
            "}"
        )
        system_prompt = get_effective_system_prompt(system_prompt, mode="process")
        if rag_context:
            system_prompt += (
                "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
                f"{rag_context}\n"
            )
            
        
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Full Requirement Context: \"{current_text}\""}
        ]
        
        response = llm.get_response(messages, stream=False)

        raw_response = response.choices[0].message.content.strip()

        try:
            data = clean_and_parse_json(raw_response)

            split_required = data.get("split_required", False)

            corrected_requirements = data.get(
                "corrected_requirements",
                []
            )

            if corrected_requirements:
                full_corrected = "\n".join(
                    req.strip()
                    for req in corrected_requirements
                    if req and req.strip()
                )
            else:
                full_corrected = current_text

        except Exception:
            # Fallback for models that return plain text
            full_corrected = raw_response

        if full_corrected.startswith("```"):
            lines = full_corrected.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].endswith("```"):
                    lines = lines[:-1]
            full_corrected = "\n".join(lines).strip()

        if (
            full_corrected.startswith('"')
            and full_corrected.endswith('"')
        ) or (
            full_corrected.startswith("'")
            and full_corrected.endswith("'")
        ):
            full_corrected = full_corrected[1:-1].strip()

        if not full_corrected:
            full_corrected = current_text
            
        return index, r, r.content, full_corrected, full_corrected
    except Exception as e:
        return index, r, r.content, f"LLM Error: {str(e)}", r.content

def correct_batch(batch_items, llm, rag, selected_collections=None):
    results = {}
    if not batch_items:
        return results

    rag_contexts = []
    if rag:
        try:
            reqs_text = [r.content for _, r in batch_items]
            rag_contexts = rag.query_batch(reqs_text, collection_name=selected_collections, top_k=2)
        except Exception:
            rag_contexts = [""] * len(batch_items)
    else:
        rag_contexts = [""] * len(batch_items)

    system_prompt = (
        "You are a strict, deterministic Senior Systems Engineer and Requirements Expert.\n"
        "Your task is to analyze and correct multiple engineering requirements using INCOSE guidelines and EARS syntax.\n"
        "You MUST adhere to these strict rules:\n"
        "1. Return ONLY valid JSON exactly matching the schema below.\n"
        "2. Do NOT include any explanation or markdown formatting outside the JSON.\n"
        "3. The output MUST be a JSON array of objects, in the EXACT same order as the inputs.\n\n"
        "JSON Schema:\n"
        "[\n"
        "  {\n"
        "    \"split_required\": boolean,\n"
        "    \"corrected_requirements\": [string]\n"
        "  }\n"
        "]"
    )

    combined_rag_context = ""
    for i, ctx in enumerate(rag_contexts):
        if ctx:
            combined_rag_context += f"Context for Requirement {i+1}:\n{ctx}\n"

    if combined_rag_context:
        system_prompt += (
            "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
            f"{combined_rag_context}\n"
        )

    system_prompt = get_effective_system_prompt(system_prompt, mode="batch_process")

    user_content = "Correct the following requirements:\n\n"
    for i, (idx, r) in enumerate(batch_items):
        user_content += f"--- Requirement {i+1} ---\n"
        user_content += f"ID: {r.name}\n"
        user_content += f"Text: \"{r.content}\"\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    try:
        response = llm.get_response(messages, stream=False)
        raw_text = response.choices[0].message.content
        data = clean_and_parse_json(raw_text)
        
        if not isinstance(data, list) or len(data) != len(batch_items):
            raise ValueError("LLM did not return an array of the correct length.")
            
        for i, (idx, r) in enumerate(batch_items):
            item_data = data[i]
            corrected_requirements = item_data.get("corrected_requirements", [])
            if corrected_requirements:
                full_corrected = "\n".join(req.strip() for req in corrected_requirements if req and req.strip())
            else:
                full_corrected = r.content
                
            if not full_corrected:
                full_corrected = r.content
                
            results[idx] = (r, r.content, full_corrected, full_corrected)
        return results
    except Exception as e:
        # Fallback to single parallel processing
        fallback_results = {}
        with ThreadPoolExecutor(max_workers=min(10, len(batch_items))) as ex:
            fs = {ex.submit(correct_single_requirement, idx, r, llm, rag, rag_contexts[i] if rag_contexts else None, selected_collections): idx for i, (idx, r) in enumerate(batch_items)}
            for f in as_completed(fs):
                idx = fs[f]
                _, r_obj, action_part, corrected_action, full_corrected = f.result()
                fallback_results[idx] = (r_obj, action_part, corrected_action, full_corrected)
        return fallback_results

def _expand_corrections(correction_data_map):
    correction_data = []
    for k in sorted(correction_data_map.keys()):
        r_info = correction_data_map[k]
        corrected_text = r_info["Corrected Requirement"]
        
        if isinstance(corrected_text, list):
            split_reqs = [str(req).strip() for req in corrected_text if str(req).strip()]
        else:
            corrected_text = str(corrected_text).replace('\\n', '\n').replace('\r', '\n')
            # Handle cases where LLM numbers them without newlines like "1. Req 2. Req"
            import re
            if '\n' not in corrected_text and re.search(r'\s+\d+\.\s+[A-Z]', corrected_text):
                corrected_text = re.sub(r'(\s+)(\d+\.\s+[A-Z])', r'\n\2', corrected_text)
                
            split_reqs = [req.strip() for req in corrected_text.split('\n') if req.strip()]
        
        for req in split_reqs:
            correction_data.append({
                "ID": r_info["ID"],
                "Original Requirement": r_info["Original Requirement"],
                "Corrected Requirement": req
            })
    return correction_data

def correct_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None, selected_collections=None, batch_size=10) -> List[Dict[str, Any]]:
    total = len(requirements)
    correction_data_map = {}
    
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return correct_batch(batch, llm, rag, selected_collections)
        except Exception:
            fallback_results = {}
            with ThreadPoolExecutor(max_workers=min(10, len(batch))) as ex:
                fs = {ex.submit(correct_single_requirement, idx, r, llm, rag, selected_collections=selected_collections): idx for idx, r in batch}
                for f in as_completed(fs):
                    idx = fs[f]
                    _, r_obj, action_part, corrected_action, full_corrected = f.result()
                    fallback_results[idx] = (r_obj, action_part, corrected_action, full_corrected)
            return fallback_results

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        
        completed_count = 0
        for future in as_completed(futures):
            batch_res = future.result()
            for idx, res in batch_res.items():
                correction_data_map[idx] = {
                    "ID": res[0].name,
                    "Original Requirement": res[0].content,
                    "Corrected Requirement": res[3]
                }
                completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total, _expand_corrections(correction_data_map))
                
    return _expand_corrections(correction_data_map)

def correct_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None, batch_size=10) -> List[Dict[str, Any]]:
    if not llm:
        raise ValueError("LLMManager is required for quality analysis.")

    total = len(requirements)
    if total == 0:
        return []

    if mode == "batch":
        return correct_requirements_batch(requirements, llm, progress_callback, rag, selected_collections, batch_size)

    rag_contexts = [""] * total
    if rag:
        try:
            full_reqs = [r.content for r in requirements]
            rag_contexts = rag.query_batch(full_reqs, collection_name=selected_collections, top_k=2)
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
            index, r_obj, action_part, corrected_action, full_corrected = future.result()
            correction_data_map[index] = {
                "ID": r_obj.name,
                "Original Requirement": r_obj.content,
                "Corrected Requirement": full_corrected
            }
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total, _expand_corrections(correction_data_map))
                
    return _expand_corrections(correction_data_map)

def generate_markdown_report(analysis_results: List[Dict[str, Any]], correction_results: List[Dict[str, Any]], file_title: str) -> str:
    md_content = f"# Compliance Report: {file_title}\n\n"
    md_content += "## Validation Issues\n\n"
    
    issues_found = False
    for r in analysis_results:
        if r.get("Status") == "Review":
            issues_found = True
            md_content += f"### ID: {r.get('ID', 'N/A')}\n"
            md_content += f"**Requirement:** {r.get('Requirement', '')}\n\n"
            md_content += f"**Failed Rule:** {r.get('Failed Rule', 'Unknown')}\n\n"
            md_content += f"**Rationale:** {r.get('Rationale', '')}\n\n"
            md_content += "---\n\n"
            
    if not issues_found:
        md_content += "No compliance issues found!\n\n"
        
    if correction_results:
        md_content += "## Automated Corrections\n\n"
        
        grouped_corrections = {}
        for cr in correction_results:
            cr_id = cr.get("ID", "N/A")
            if cr_id not in grouped_corrections:
                grouped_corrections[cr_id] = {
                    "Original": cr.get("Original Requirement", ""),
                    "Corrected": []
                }
            grouped_corrections[cr_id]["Corrected"].append(cr.get("Corrected Requirement", ""))
            
        corrections_found = False
        for cr_id, data in grouped_corrections.items():
            orig = data["Original"]
            corrected_list = data["Corrected"]
            
            if len(corrected_list) > 1 or (len(corrected_list) == 1 and orig != corrected_list[0]):
                corrections_found = True
                md_content += f"### ID: {cr_id}\n"
                md_content += f"**Original:** {orig}\n\n"
                md_content += "**Corrected:**\n"
                for c in corrected_list:
                    md_content += f"- {c}\n"
                md_content += "\n---\n\n"
        if not corrections_found:
            md_content += "No corrections needed!\n\n"
            
    return md_content
