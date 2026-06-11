import os
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Prioritize api_key.env fallback mapping
#env_path = Path(".env") if Path(".env").exists() else Path("api_key.env")
#load_dotenv(env_path)

class LLMManager:
    def __init__(self, model_name="nvidia/llama-3.3-nemotron-super-49b-v1.5"):
        #api_key = os.getenv("NVIDIA_API_KEY")
        api_key = st.secrets["API_KEY"]
        if not api_key:
            raise EnvironmentError("NVIDIA_API_KEY is not set. Add it to .env or api_key.env in the project root.")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model_name = model_name
        self.embedding_model = "nvidia/embeddings-nv-embed-qa-4"

    def _retry_api_call(self, api_func, *args, **kwargs):
        """Helper to execute API functions with transient failure retry logic using exponential backoff."""
        try:
            import streamlit as st
            retries = st.session_state.get("llm_retries", 3)
        except (ImportError, AttributeError):
            retries = 3

        import time
        last_exception = None
        for attempt in range(retries + 1):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < retries:
                    wait_time = 2 ** attempt
                    print(f"[LLMManager] API call failed: {e}. Retrying {attempt+1}/{retries} in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"[LLMManager] API call failed after {retries} retries: {e}", flush=True)
                    raise last_exception

    def get_response(self, messages, stream=True):
        # Keep a sliding window of the last 10 messages to avoid token limit issues
        trimmed_messages = messages[-10:]
        return self._retry_api_call(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=trimmed_messages,
            temperature=0.2,  # Fixed low for deterministic analysis stability
            top_p=1,
            max_tokens=2048,
            stream=stream
        )

    def get_embedding(self, text: str):
        """Generates contextual float embeddings using the NVIDIA NIM footprint"""
        response = self._retry_api_call(
            self.client.embeddings.create,
            input=[text],
            model=self.embedding_model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]):
        """Generates embeddings for a list of strings in a single batch API call."""
        if not texts:
            return []
        response = self._retry_api_call(
            self.client.embeddings.create,
            input=texts,
            model=self.embedding_model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        # Ensure correct ordering by sorting on the index property
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
