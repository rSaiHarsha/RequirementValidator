import os
from Model.requirement import Requirement
from RagEngine.rag_engine import RAGEngine
from Analysis.quality_analyser import correct_single_requirement, analyze_single_requirement
import sys

# Import the actual LLM from app or wherever
# Actually, let's just initialize LLMManager
from Model.llm_manager import LLMManager

def main():
    llm = LLMManager()
    rag = None
    
    # A requirement with multiple actions
    req = Requirement(name="REQ-1", content="If raining, the system shall measure temperature and calculate the pressure.", rationale="")
    
    # 1. First test analysis
    print("--- Original Analysis ---")
    _, a_res = analyze_single_requirement(0, req, llm, rag)
    print(a_res)
    
    # 2. Test correction
    print("\n--- Correction ---")
    index, r, original, corrected, final_corrected = correct_single_requirement(0, req, llm, rag)
    print("FINAL CORRECTED:\n" + final_corrected)

if __name__ == "__main__":
    main()
