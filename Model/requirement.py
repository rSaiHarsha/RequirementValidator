import csv
import json
from typing import List

class Requirement:
    """
    A simple class representing a Requirement.
    """
    def __init__(self, name: str, content: str, state: str, asil: str, rationale: str, covers: str = "", refined: str = ""):
        self.name = name
        self.content = content
        self.state = state
        self.asil = asil
        self.rationale = rationale
        self.covers = covers
        self.refined = refined

    def to_dict(self):
        """Convert object to standard CSV row dictionary."""
        return {
            "Name": self.name,
            "Content": self.content,
            "State": self.state,
            "ASIL": self.asil,
            "Rationale": self.rationale,
            "Covers": self.covers,
            "Refined": self.refined
        }

    @staticmethod
    def load_from_csv(file_path: str) -> List["Requirement"]:
        """Load requirements from a CSV file into objects with dynamic header matching."""
        requirements = []
        with open(file_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Standardize keys
            headers = reader.fieldnames if reader.fieldnames else []
            
            # Helper to find matching header dynamically
            def find_header(possible_names, default):
                for h in headers:
                    if h and any(name in h.lower() for name in possible_names):
                        return h
                return default

            name_key = find_header(["name"], "Name")
            content_key = find_header(["content", "requirement", "text"], "Content")
            state_key = find_header(["state", "status"], "State")
            asil_key = find_header(["asil", "severity"], "ASIL")
            rationale_key = find_header(["rationale", "reason", "description"], "Rationale")
            covers_key = find_header(["covers"], "Covers(int)")
            refined_key = find_header(["refined"], "Refined(int)")

            for row in reader:
                requirements.append(Requirement(
                    name=row.get(name_key, "") or "",
                    content=row.get(content_key, "") or "",
                    state=row.get(state_key, "") or "",
                    asil=row.get(asil_key, "") or "",
                    rationale=row.get(rationale_key, "") or "",
                    covers=row.get(covers_key, "") or "",
                    refined=row.get(refined_key, "") or ""
                ))
        return requirements

    @staticmethod
    def save_to_csv(requirements: List["Requirement"], file_path: str):
        """Save a list of Requirement objects back to a CSV file."""
        headers = ["Name", "Content", "State", "ASIL", "Rationale", "Covers(int)", "Refined(int)"]
        with open(file_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for req in requirements:
                writer.writerow(req.to_dict())

    # Easiest way to store/retrieve for AI LLMs (simple JSON list representation)
    def to_json(self) -> str:
        """Convert object to JSON string (great for LLM prompts or APIs)."""
        return json.dumps(self.to_dict(), indent=2)
