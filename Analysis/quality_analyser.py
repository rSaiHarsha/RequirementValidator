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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*\]', ']', text)
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON even after cleanup: {str(e)}")

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
            "   - MUST NOT combine multiple distinct requirements in a single sentence (e.g. multiple actions). (EXCEPTION: If the text contains multiple distinct requirement sentences clearly separated by newlines, this is ACCEPTABLE as long as each sentence is individually compliant and atomic).\n"
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
        "You are an expert systems engineering auditor specializing in ASPICE, INCOSE, and EARS requirement standards.\n"
        "Your task is to structurally parse and analyze a batch of engineering requirements.\n"
        "1. EARS Syntax (Preconditions/Triggers): If the requirement has preconditions or triggers, they must start with EARS keywords (If, When, While, Where).\n"
        "   (Note: If the precondition violates EARS syntax but the action part is flawless, flag it as a warning in your rationale but keep the Status as 'Passed'. If severely malformed, flag 'Review').\n"
        "2. INCOSE Rules (System Response / Action Part):\n"
        "   - MUST use the standard modal verb 'shall' (do not use should, will, must, or behaves).\n"
        "   - MUST NOT contain vague or subjective terms (e.g. fast, quickly, beautifully, great, modern, creepy).\n"
        "   - MUST be verifiable and measurable.\n"
        "   - MUST NOT combine multiple distinct requirements in a single sentence (e.g. multiple actions). (EXCEPTION: If the text contains multiple distinct requirement sentences clearly separated by newlines, this is ACCEPTABLE as long as each sentence is individually compliant and atomic).\n"
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
        {"role": "user", "content":  f"""
Requirement:{user_content}Analyze the requirement.
Correct INCOSE violations.
Correct EARS violations.
Split if multiple behaviors exist.
Return JSON only."""}
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
    
    batch_size = 10
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return analyze_batch(batch, llm, rag, selected_collections)
        except Exception:
            fallback_results = {}
            for idx, r in batch:
                _, res = analyze_single_requirement(idx, r, llm, rag, selected_collections=selected_collections)
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

def analyze_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None) -> List[Dict[str, Any]]:
    if not llm:
        raise ValueError("LLMManager is required for quality analysis.")

    total = len(requirements)
    if total == 0:
        return []

    if mode == "batch":
        return analyze_requirements_batch(requirements, llm, progress_callback, rag, selected_collections)

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
    
        for attempt in range(max_retries):
            system_prompt = """
You are a Senior Systems Engineer, INCOSE Requirements Expert,
EARS Expert, ASPICE Assessor, and ISO 26262 Functional Safety Engineer.

Your task is to review and correct engineering requirements.

MANDATORY RULES

1. EARS Compliance
- Determine the correct EARS pattern:
  - Ubiquitous
  - Event Driven
  - State Driven
  - Optional Feature
  - Unwanted Behavior
  - Complex
- Rewrite the requirement using the appropriate EARS syntax.
- Use standard EARS keywords only:
  - WHEN
  - WHILE
  - IF ... THEN
  - WHERE

2. Modal Verb
- Use SHALL as the only modal verb.
- Replace:
  - should
  - will
  - must
  - can
  - may
  - behaves
with SHALL where appropriate.

3. INCOSE Compliance
Ensure each requirement is:
- Necessary
- Correct
- Unambiguous
- Complete
- Singular
- Feasible
- Verifiable
- Consistent
- Traceable
- Implementation Free

4. Atomic Requirement Rule
A requirement shall contain EXACTLY ONE system behavior.

ONE SHALL = ONE REQUIREMENT.

Split the requirement if:
- Multiple actions occur after SHALL.
- Multiple verbs are connected by AND.
- Multiple verbs are connected by OR.
- Multiple actions are separated by commas.
- Detection and reaction are combined.
- Calculation and transmission are combined.
- Monitoring and reporting are combined.
- Fault detection and fault handling are combined.
- Logging and notification are combined.

When splitting:
- Repeat the exact same trigger/precondition.
- Preserve all timing constraints.
- Preserve all fault identifiers.
- Preserve all DTC references.
- Preserve all safety intent.
- Preserve ASIL-related information.
- Preserve thresholds and units.

5. Ambiguity Removal
Replace vague words such as:
- appropriate
- sufficient
- normal
- quickly
- efficiently
- robust
- user-friendly
- timely
- optimized
- safe

with specific and measurable criteria whenever possible.

6. Preservation Rules
DO NOT:
- Add new functionality.
- Invent values.
- Change thresholds.
- Change timing constraints.
- Change fault IDs.
- Change DTC identifiers.
- Change safety intent.

ONLY correct violations.

7. Self-Validation Before Returning
Verify:
- Every requirement contains SHALL.
- Every requirement follows EARS syntax.
- Every requirement contains exactly one behavior.
- No ambiguous terms remain.
- Timing constraints are preserved.
- Units are preserved.
- Safety intent is preserved.

OUTPUT FORMAT

Return JSON only.

{
  "split_required": true,
  "corrected_requirements": [
    "WHEN condition, THE SYSTEM SHALL perform action A.",
    "WHEN condition, THE SYSTEM SHALL perform action B."
  ]
}

If no correction is needed:

{
  "split_required": false,
  "corrected_requirements": [
    "<original requirement exactly as provided>"
  ]
}

Do not include explanations.
Do not include markdown.
Do not include code fences.
Return valid JSON only.
"""

            system_prompt += """

EXAMPLE

Input:

IF VehSpd AliveCounter is unchanged for [3 consecutive cycles],
THEN THE SYSTEM SHALL:
- detect F_ALC_VehSpd within [10 ms]
- react within [10 ms]
- set DTC_101_VSPD_ALC
- transition to safe state.

Output:

{
  "split_required": true,
  "corrected_requirements": [
    "IF VehSpd AliveCounter is unchanged for [3 consecutive cycles], THEN THE SYSTEM SHALL detect F_ALC_VehSpd within [10 ms].",
    "IF VehSpd AliveCounter is unchanged for [3 consecutive cycles], THEN THE SYSTEM SHALL react within [10 ms].",
    "IF VehSpd AliveCounter is unchanged for [3 consecutive cycles], THEN THE SYSTEM SHALL set DTC_101_VSPD_ALC.",
    "IF VehSpd AliveCounter is unchanged for [3 consecutive cycles], THEN THE SYSTEM SHALL transition to safe state."
  ]
}
"""
            if rag_context:
                system_prompt += (
                    "\nIn addition to standard rules, you MUST also conform to these project-specific rules retrieved from the knowledge base:\n"
                    f"{rag_context}\n"
                )
            
            
            if failed_rule and rationale:
                system_prompt += (
                    f"\n\nIMPORTANT FEEDBACK ON PREVIOUS ATTEMPT:\n"
                    f"Your previous correction failed the '{failed_rule}' rule.\n"
                    f"Rationale for failure: {rationale}\n"
                    "Please fix this specific issue and ensure all rules are followed."
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
                
            temp_r = Requirement(name=r.name, content=full_corrected, rationale=r.rationale)
            temp_r.state = r.state
            temp_r.asil = r.asil
            
            _, analysis_res = analyze_single_requirement(index, temp_r, llm, rag, rag_context, selected_collections)
            
            if analysis_res.get("Status") == "Passed" or attempt == max_retries - 1:
                return index, r, r.content, full_corrected, full_corrected
                
            failed_rule = analysis_res.get("Failed Rule", "Unknown")
            rationale = analysis_res.get("Rationale", "Still violates rules")
            current_text = full_corrected
            
        return index, r, r.content, current_text, current_text
    except Exception as e:
        return index, r, r.content, f"LLM Error: {str(e)}", r.content

def correct_batch(batch_items, llm, rag, selected_collections=None):
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
        "4. Atomic Requirements: A requirement MUST NOT combine multiple distinct actions. If a requirement contains multiple distinct actions, you MUST split it into multiple distinct requirements. Each split requirement MUST repeat the exact same precondition/trigger from the original requirement. Separate them with a single newline (do NOT use bullet points).\n"
        "5. If the requirement is already compliant, keep it EXACTLY as-is.\n\n"
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
            
    # Verify the batch corrections
    verify_items = []
    for idx, (r_obj, original_text, action_part, full_corrected) in batch_results.items():
        temp_r = Requirement(name=r_obj.name, content=full_corrected, rationale=r_obj.rationale)
        temp_r.state = r_obj.state
        temp_r.asil = r_obj.asil
        verify_items.append((idx, temp_r))
        
    try:
        verify_res = analyze_batch(verify_items, llm, rag, selected_collections)
    except Exception:
        verify_res = {}
        
    for idx, (r_obj, original_text, action_part, full_corrected) in batch_results.items():
        v_res = verify_res.get(idx, {})
        if v_res.get("Status") == "Review":
            _, _, _, _, retried_corrected = correct_single_requirement(
                idx, r_obj, llm, rag, None, selected_collections,
                feedback_rule=v_res.get("Failed Rule"),
                feedback_rationale=v_res.get("Rationale"),
                initial_text=full_corrected
            )
            batch_results[idx] = (r_obj, original_text, action_part, retried_corrected)
            
    return batch_results

def _expand_corrections(correction_data_map):
    correction_data = []
    for k in sorted(correction_data_map.keys()):
        r_info = correction_data_map[k]
        corrected_text = r_info["Corrected Requirement"]
        
        split_reqs = [req.strip() for req in corrected_text.split('\n') if req.strip()]
        
        for i, split_req in enumerate(split_reqs):
            new_id = r_info["ID"] if len(split_reqs) == 1 else f"{r_info['ID']}.{i+1}"
            correction_data.append({
                "ID": new_id,
                "Original Requirement": r_info["Original Requirement"],
                "Corrected Requirement": split_req
            })
    return correction_data

def correct_requirements_batch(requirements: List[Requirement], llm, progress_callback=None, rag=None, selected_collections=None) -> List[Dict[str, Any]]:
    total = len(requirements)
    correction_data_map = {}
    
    batch_size = 10
    batches = []
    for i in range(0, total, batch_size):
        batches.append([(idx, requirements[idx]) for idx in range(i, min(i + batch_size, total))])
        
    def process_batch(batch):
        try:
            return correct_batch(batch, llm, rag, selected_collections)
        except Exception:
            fallback_results = {}
            for idx, r in batch:
                _, r_obj, action_part, corrected_action, full_corrected = correct_single_requirement(idx, r, llm, rag, selected_collections=selected_collections)
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

def correct_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None) -> List[Dict[str, Any]]:
    if not llm:
        raise ValueError("LLMManager is required for quality analysis.")

    total = len(requirements)
    if total == 0:
        return []

    if mode == "batch":
        return correct_requirements_batch(requirements, llm, progress_callback, rag, selected_collections)

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
        corrections_found = False
        for cr in correction_results:
            if cr.get("Original Requirement") != cr.get("Corrected Requirement"):
                corrections_found = True
                md_content += f"### ID: {cr.get('ID', 'N/A')}\n"
                md_content += f"**Original:** {cr.get('Original Requirement', '')}\n\n"
                md_content += f"**Corrected:** {cr.get('Corrected Requirement', '')}\n\n"
                md_content += "---\n\n"
        if not corrections_found:
            md_content += "No corrections needed!\n\n"
            
    return md_content
