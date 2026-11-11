"""One client for every OpenAI-compatible endpoint (Gemini, Groq, Mistral, OpenAI, Ollama), with a disk cache
and a rate limiter so that free tiers are enough and a rerun costs nothing."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

BACKENDS = {  # base URL, default model, environment variable holding the key
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "models/gemini-3.8-flash", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", None),
}
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "llm"


class LLM:
    def __init__(self, backend: str, model: str | None = None, rpm: float = 12, temperature: float = 0.2):
        base, default_model, env = BACKENDS[backend]
        self.base, self.model, self.temperature = base, model or default_model, temperature
        self.key = os.environ.get(env, "") if env else "ollama"
        if env and not self.key:
            raise SystemExit(f"Set the {env} environment variable (never put the key in the code).")
        self.local = backend == "ollama"
        self.timeout = 900 if self.local else 120     # a local model on CPU can take minutes per answer
        self.min_gap = 60.0 / rpm
        self.last = 0.0
        self.calls = self.cached = self.tokens = 0
        CACHE.mkdir(parents=True, exist_ok=True)

    def chat(self, messages: list[dict], tag: str = "") -> str:
        key = hashlib.sha256(json.dumps([self.model, self.temperature, tag, messages]).encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text())["text"]
        for attempt in range(8):
            wait = self.min_gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": 700}
            try:
                r = requests.post(f"{self.base}/chat/completions", timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"}, json=body)
            except (requests.Timeout, requests.ConnectionError) as e:
                if self.local and isinstance(e, requests.ConnectionError) and attempt == 0:
                    print("  cannot reach Ollama: is it running? (open the Ollama app or run 'ollama serve')",
                          flush=True)
                print(f"  {type(e).__name__}, retrying (attempt {attempt + 1}/8)", flush=True)
                time.sleep(min(60, 10 * (attempt + 1)))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                body = r.text
                if r.status_code == 429 and re.search(r"PerDay|per day|daily", body, re.I):
                    raise SystemExit(f"Daily quota of {self.model} is used up. The answers already received are "
                                     f"cached: rerun tomorrow or with another model and the run resumes.\n{body[:600]}")
                delay = re.search(r'"retryDelay":\s*"(\d+)', body)           # Gemini says how long to wait
                pause = int(delay.group(1)) + 2 if delay else min(90, 10 * 2 ** attempt)
                print(f"  HTTP {r.status_code}, waiting {pause} s (attempt {attempt + 1}/8)", flush=True)
                time.sleep(pause)
                continue
            if r.status_code >= 400:                     # show the provider's message (wrong model name, quota...)
                raise SystemExit(f"HTTP {r.status_code} from {self.base} for model {self.model}:\n{r.text[:800]}")
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            self.calls += 1
            self.tokens += int(data.get("usage", {}).get("total_tokens", 0))
            path.write_text(json.dumps({"text": text}))
            return text
        raise RuntimeError("LLM endpoint kept failing (rate limit, server error or timeout); rerun to resume from the cache")


def parse_json(text: str) -> dict | None:
    """Models wrap JSON in prose or code fences; take the first object that parses."""
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    return json.loads(text[m.start(): j + 1])
                except json.JSONDecodeError:
                    break
    return None
