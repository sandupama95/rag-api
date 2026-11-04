import os

import requests


class OllamaClient:
    def generate(self, prompt: str) -> str:
        r = requests.post(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
            json={"model": os.getenv("MODEL_NAME", "llama3.2:3b"),
                  "prompt": prompt, "stream": False},
            timeout=120,
        )
        return r.json()["response"]

class GroqClient:
    def generate(self, prompt: str) -> str:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("MODEL_NAME", "qwen/qwen3.8-27b"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "400")),
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def get_llm():
    backend = os.getenv("LLM_BACKEND", "ollama")   # "ollama" or "groq"
    return GroqClient() if backend == "groq" else OllamaClient()