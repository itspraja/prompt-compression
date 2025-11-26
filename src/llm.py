"""
llm.py
Optional LLM inference helpers. Supports:
- OpenAI (if OPENAI_API_KEY set)
- HuggingFace Inference API (if HF_API_TOKEN set)
- Local transformers fallback for small models (optional)
"""

import os
import requests
import json
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_INFERENCE_URL = os.getenv("HF_INFERENCE_API_URL", "https://api-inference.huggingface.co/models/")

def call_openai_completion(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Minimal OpenAI chat completion call (if key is present).
    Note: this function is intentionally simple — for heavy use prefer the official library.
    """
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def call_hf_inference(prompt: str, model: str = "google/flan-t5-small", max_tokens: int = 256) -> str:
    """
    Calls HuggingFace Inference API if HF_TOKEN set. Model name is appended to HF_INFERENCE_API_URL.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in environment.")
    url = HF_INFERENCE_URL + model
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "options": {"use_cache": False, "wait_for_model": True}, "parameters": {"max_new_tokens": max_tokens}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # HF returns a list or dict; try to extract text
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF inference error: {data['error']}")
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # fallback:
    if isinstance(data, str):
        return data
    # best-effort extraction
    return json.dumps(data)

# Local fallback using transformers (small models)
def call_local_transformer(prompt: str, model_name: str = "google/flan-t5-small", max_tokens: int = 128) -> str:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tok(prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError("Local transformer call failed: " + str(e))

def infer(prompt: str, backend: str = "auto", model: str = None, max_tokens: int = 128, temperature: float = 0.0) -> str:
    """
    backend: 'openai', 'hf', 'local', 'auto'
    model: backend-specific model id
    """
    if backend == "openai":
        return call_openai_completion(prompt, model or "gpt-3.5-turbo", max_tokens=max_tokens, temperature=temperature)
    if backend == "hf":
        return call_hf_inference(prompt, model or "google/flan-t5-small", max_tokens=max_tokens)
    if backend == "local":
        return call_local_transformer(prompt, model or "google/flan-t5-small", max_tokens=max_tokens)
    # auto: pick available
    if OPENAI_KEY:
        return call_openai_completion(prompt, model or "gpt-3.5-turbo", max_tokens=max_tokens, temperature=temperature)
    if HF_TOKEN:
        return call_hf_inference(prompt, model or "google/flan-t5-small", max_tokens=max_tokens)
    # fallback local
    return call_local_transformer(prompt, model or "google/flan-t5-small", max_tokens=max_tokens)
