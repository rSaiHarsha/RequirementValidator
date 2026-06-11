import os
import tempfile
from typing import List
from Model.requirement import Requirement

def load_uploaded_requirements(uploaded_file) -> List[Requirement]:
    """
    Saves an uploaded Streamlit file temporarily and loads it using the Requirement class.
    """
    if uploaded_file is None:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        reqs = Requirement.load_from_csv(tmp_path)
    finally:
        os.remove(tmp_path)
    return reqs
