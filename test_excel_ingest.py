import io
import os
import uuid
import sys

# Ensure the correct import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RagEngine.rag_engine import RAGEngine
from Model.llm import LLMManager

class MockLLMManager(LLMManager):
    def __init__(self):
        # Prevent actual api key initialization issues in testing
        pass
    def get_embedding(self, text):
        return [0.0] * 1024
    def get_embeddings_batch(self, texts):
        return [[0.0] * 1024 for _ in texts]

def main():
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        print("[-] Dependencies 'pandas' and 'openpyxl' are required for testing.")
        print("[-] Please run: pip install pandas openpyxl")
        return

    print("[*] Creating a temporary Excel file in-memory...")
    # Create sample requirements dataframe
    data = {
        "Requirement ID": ["REQ-001", "REQ-002", "REQ-003"],
        "Functional Area": ["LKA", "CAN", "ADAS"],
        "Description": [
            "The LKA controller shall transition to Active state when lane departure is detected.",
            "The system shall transmit CAN message CAN_LKA_Trig every 10ms.",
            "The vehicle shall alert the driver if lane markings are not visible."
        ]
    }
    df = pd.DataFrame(data)
    
    # Save to a BytesIO object to simulate an uploaded file
    excel_bytes_io = io.BytesIO()
    with pd.ExcelWriter(excel_bytes_io, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    
    excel_bytes = excel_bytes_io.getvalue()
    
    # Initialize RAG Engine with Mock LLM
    print("[*] Initializing RAG Engine...")
    llm = MockLLMManager()
    rag = RAGEngine(llm_manager=llm)
    
    # Ingest the Excel bytes
    print("[*] Processing Excel bytes via RAGEngine.process_file...")
    rag.process_file("test_requirements.xlsx", excel_bytes, collection_name="test_collection")
    
    # Print the resulting parsed documents
    print(f"\n[+] Processing complete. Found {len(rag.documents)} parsed chunks:")
    for idx, doc in enumerate(rag.documents):
        print(f"\n--- Document {idx + 1} ---")
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Source: {doc['source']}")
        print(f"Collection: {doc['collection']}")
        print(f"Metadata: {doc['metadata']}")
        print(f"Text Content:\n{doc['text']}")
    
    print("\n[+] Verification successful!")

if __name__ == "__main__":
    main()
