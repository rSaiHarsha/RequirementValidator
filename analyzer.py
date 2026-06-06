import json
import re
import time
from pydantic import BaseModel, Field
from typing import List
from models import LLMManager

# --- ULTRA-LEAN PYDANTIC SCHEMA ---
# We remove the summary calculation block from the LLM to save tokens and processing time.
class DetailedReportItem(BaseModel):
    original: str = Field(description="The exact original requirement text passed.")
    status: str = Field(description="Strictly 'Good' or 'Bad'.")
    failed_rule: str = Field(description="Max 3 words naming the broken rule (e.g. 'Ambiguity'), or 'None'.")
    corrected: str = Field(description="The rewritten compliant requirement statement. Keep it brief.")

class RequirementAnalysisSchema(BaseModel):
    detailed_report: List[DetailedReportItem]


class RequirementAnalyzer:
    def __init__(self, llm_manager: LLMManager, batch_size=3): 
        self.llm = llm_manager
        self.batch_size = batch_size

    def parse_requirements(self, file_name, file_content):
        text = file_content.decode("utf-8", errors="ignore")
        cleaned_reqs = []
        
        if file_name.lower().endswith('.csv'):
            import csv, io
            try:
                csv_file = io.StringIO(text)
                reader = csv.reader(csv_file)
                headers = [h.strip().lower() for h in next(reader, [])]
                
                target_keywords = ["description", "requirement", "text", "specification", "statement"]
                target_idx = -1
                for kw in target_keywords:
                    for idx, header in enumerate(headers):
                        if kw in header:
                            target_idx = idx
                            break
                    if target_idx != -1: break
                
                if target_idx == -1 and headers:
                    target_idx = 0 
                
                for row in reader:
                    if row and len(row) > target_idx:
                        req_text = row[target_idx].strip()
                        req_text = re.sub(r'^[-*+•]\s+|^[0-9]+[\.\)]\s+', '', req_text).strip()
                        if len(req_text) > 10:
                            cleaned_reqs.append(req_text)
                return cleaned_reqs
            except Exception:
                pass

        raw_lines = text.split("\n")
        header_keywords = ["requirement id", "requirement name", "upstream traceability", "asil rating", "verification method"]
        
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"): continue
            if any(kw in line.lower() for kw in header_keywords): continue
            line = re.sub(r'^[-*+•]\s+|^[0-9]+[\.\)]\s+', '', line).strip()
            if len(line) > 10:
                cleaned_reqs.append(line)
                
        return cleaned_reqs

    def analyze(self, requirements, context_data=""):
        if not requirements:
            return {
                "summary": { "good_count": 0, "bad_count": 0, "failed_incose_rules": 0, "failed_aspice_rules": 0 },
                "detailed_report": []
            }

        combined_report = {
            "summary": { "good_count": 0, "bad_count": 0, "failed_incose_rules": 0, "failed_aspice_rules": 0 },
            "detailed_report": []
        }

        batches = [requirements[i:i + self.batch_size] for i in range(0, len(requirements), self.batch_size)]
        
        for index, batch in enumerate(batches):
            system_prompt = f"""You are an expert Automotive Quality Engineer specializing in INCOSE and ASPICE compliance.
            Evaluate the provided requirements array against strict quality guidelines:
            - Look for Ambiguity, lack of Verifiability, or weak Traceability.
            - If a requirement is compliant, set status to 'Good', failed_rule to 'None', and copy the original text to 'corrected'.
            - If non-compliant, set status to 'Bad', name the rule broken, and provide the rewrite.

            Contextual RAG Data for reference lookup:
            {context_data}

            CRITICAL: Be concise. Do NOT include explanations or introductory text anywhere."""
            user_payload = f"Requirements to evaluate:\n" + "\n".join([f"- {r}" for r in batch])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ]
            
            try:
                # Restored max_tokens to 2048 to guarantee safety headroom against cutoffs
                response = self.llm.client.beta.chat.completions.parse(
                    model=self.llm.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048, 
                    response_format=RequirementAnalysisSchema,
                )
                
                if response is None or not response.choices:
                    raise ValueError("Empty endpoint object response received.")
                
                parsed_object = response.choices[0].message.parsed
                if not parsed_object:
                    raise ValueError("Model content failed schema validation requirements.")
                
                batch_json = parsed_object.model_dump()
                reports = batch_json.get("detailed_report", [])
                
                # --- PYTHON LOCAL METRICS CALCULATION ---
                # Instead of forcing the LLM to count, we compute metrics locally via Python loops.
                # This guarantees accuracy and prevents JSON truncation errors.
                for item in reports:
                    if item["status"].strip().lower() == "good":
                        combined_report["summary"]["good_count"] += 1
                    else:
                        combined_report["summary"]["bad_count"] += 1
                        
                        # Match rules to update telemetry metrics categories dynamically
                        rule = item["failed_rule"].lower()
                        if "incose" in rule or "ambiguity" in rule or "verify" in rule:
                            combined_report["summary"]["failed_incose_rules"] += 1
                        elif "aspice" in rule or "trace" in rule:
                            combined_report["summary"]["failed_aspice_rules"] += 1
                        else:
                            # Default category increment split fallback
                            combined_report["summary"]["failed_incose_rules"] += 1
                            
                    combined_report["detailed_report"].append(item)
                
            except Exception as e:
                error_label = f"Truncation/Parsing Error: {str(e)[:40]}"
                combined_report["summary"]["bad_count"] += len(batch)
                combined_report["summary"]["failed_incose_rules"] += len(batch)
                
                for r in batch:
                    combined_report["detailed_report"].append({
                        "original": r, 
                        "status": "Bad", 
                        "failed_rule": "Token Limit Timeout", 
                        "corrected": f"Skipped block protection triggered. Context: {error_label}"
                    })
            
            time.sleep(0.1)

        return combined_report