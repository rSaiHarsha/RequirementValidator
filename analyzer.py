import json
import re
from models import LLMManager

class RequirementAnalyzer:
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    def parse_requirements(self, file_content):
        text = file_content.decode("utf-8", errors="ignore")
        raw_lines = text.split("\n")
        cleaned_reqs = []
        
        # Keywords commonly found in CSV/Table headers to ignore
        header_keywords = ["requirement id", "requirement name", "upstream traceability", "asil rating", "verification method"]
        
        for line in raw_lines:
            line = line.strip()
            
            # Skip empty lines, technical markdown headers, or divider lines
            if not line or line.startswith("#") or line.startswith("---"):
                continue
                
            # FIX 1: Ignore header rows dynamically
            if any(kw in line.lower() for kw in header_keywords):
                continue
                
            # Clean off Markdown bullet syntax layout junk
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

        system_prompt = f"""You are an expert Systems and Automotive Quality Engineer specializing in INCOSE and ASPICE compliance.
Analyze the provided user requirements against strict engineering principles:
1. Ambiguity (Avoid 'TBD', 'easy', 'fast', 'user-friendly')
2. Verifiability (Must be quantitatively measurable)
3. Traceability (ASPICE compliance)

Contextual RAG Data provided for lookup reference:
{context_data}

You MUST strictly output a JSON object matching this schema perfectly:
{{
    "summary": {{ "good_count": 2, "bad_count": 1, "failed_incose_rules": 1, "failed_aspice_rules": 0 }},
    "detailed_report": [
         {{ "original": "The system should be fast.", "status": "Bad", "failed_rule": "INCOSE Ambiguity", "corrected": "The system processing latency shall be less than 50ms." }}
    ]
}}
"""
        user_payload = "Requirements to assess:\n" + "\n".join([f"- {r}" for r in requirements])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]
        
        try:
            # FIX 2: Enforce JSON mode natively in the API call
            response = self.llm.client.chat.completions.create(
                model=self.llm.model_name,
                messages=messages,
                temperature=0.1, 
                max_tokens=2048,
                response_format={"type": "json_object"} # Forces strict JSON generation
            )
            
            raw_text = response.choices[0].message.content
            if not raw_text:
                raise ValueError("Empty response text received.")

            return json.loads(raw_text.strip())
            
        except Exception as e:
            # Enhanced debugging fallback that displays the actual error in the table
            error_msg = f"Parser Error: {str(e)[:50]}"
            return {
                "summary": {"good_count": 0, "bad_count": len(requirements), "failed_incose_rules": len(requirements), "failed_aspice_rules": 0},
                "detailed_report": [{"original": r, "status": "Bad", "failed_rule": error_msg, "corrected": "Ensure text contains descriptive system behavior statements."} for r in requirements]
            }