import json
import re
from typing import List, Dict, Any
from Model.llm import LLMManager
from Model.requirement import Requirement

# Import modular backend logic
from Analysis.quality_analyser import analyze_requirements, correct_requirements, generate_markdown_report
from Analysis.traceability_analyser import compare_traceability
from Analysis.diagram_aligner import compare_hld_alignment, compare_lld_alignment

class RequirementAnalyzer:
    def __init__(self, llm_manager: LLMManager): 
        self.llm = llm_manager

    # --- REFACTORED DELEGATE METHODS ---

    def analyze_requirements(self, requirements: List[Requirement], progress_callback=None, rag=None, mode="single", selected_collections=None, batch_size=10) -> List[Dict[str, Any]]:
        return analyze_requirements(requirements, self.llm, progress_callback, rag, mode, selected_collections, batch_size)

    def correct_requirements(self, requirements: List[Requirement], progress_callback=None, rag=None, mode="single", selected_collections=None, batch_size=10) -> List[Dict[str, Any]]:
        return correct_requirements(requirements, self.llm, progress_callback, rag, mode, selected_collections, batch_size)

    def compare_traceability(self, swe1_reqs: List[Requirement], swe2_reqs: List[Requirement]) -> Dict[str, Any]:
        return compare_traceability(swe1_reqs, swe2_reqs)

    def compare_hld_alignment(self, swe1_reqs: List[Requirement], components: List[str]) -> List[Dict[str, Any]]:
        return compare_hld_alignment(swe1_reqs, components)

    def compare_lld_alignment(self, swe2_reqs: List[Requirement], methods: List[str]) -> List[Dict[str, Any]]:
        return compare_lld_alignment(swe2_reqs, methods)

    def generate_report(self, analysis_results: List[Dict[str, Any]], correction_results: List[Dict[str, Any]], file_title: str) -> str:
        return generate_markdown_report(analysis_results, correction_results, file_title)
