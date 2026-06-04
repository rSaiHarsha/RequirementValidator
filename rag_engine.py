import os
import numpy as np
import pickle
from models import LLMManager

class RAGEngine:
    def __init__(self, llm_manager: LLMManager, db_path="vector_store.pkl"):
        self.llm = llm_manager
        self.db_path = db_path
        self.documents = []  
        self.vectors = []    

    def clear_database(self):
        self.documents = []
        self.vectors = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def process_file(self, file_name, file_content):
        text = file_content.decode("utf-8", errors="ignore")
        # Split blocks cleanly by double newline breaks
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
        
        for chunk in chunks:
            self.documents.append({
                "source": file_name,
                "text": chunk
            })

    def train_engine(self, progress_callback):
        if not self.documents:
            return False, "No documents loaded to train on."
        
        self.vectors = []
        total = len(self.documents)
        
        for i, doc in enumerate(self.documents):
            embedding = self.llm.get_embedding(doc["text"])
            self.vectors.append(embedding)
            progress_callback((i + 1) / total)
        
        with open(self.db_path, "wb") as f:
            pickle.dump({"docs": self.documents, "vecs": self.vectors}, f)
            
        return True, f"Successfully trained and indexed {total} knowledge blocks!"

    def load_trained_engine(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["docs"]
                    self.vectors = data["vecs"]
                return True
            except Exception:
                return False  # Soft reset on file corruptions
        return False

    def query(self, search_text, top_k=3):
        if not self.vectors or len(self.vectors) == 0:
            return ""
            
        query_vector = np.array(self.llm.get_embedding(search_text))
        matrix = np.array(self.vectors)
        
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return ""
            
        # Hardened calculation using 1e-8 epsilon adjustment against division anomalies
        scores = np.dot(matrix, query_vector) / (matrix_norms * query_norm + 1e-8)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        context_blocks = [self.documents[idx]["text"] for idx in top_indices if scores[idx] > 0.3]
        return "\n\n--- Context Block ---\n".join(context_blocks)