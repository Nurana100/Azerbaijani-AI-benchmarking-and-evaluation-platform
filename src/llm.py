import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

# Same models as run_benchmark.py
MODELS = {
    "Gemini (gemini-3.1-flash-lite)": ("gemini", "gemini-3.1-flash-lite"),
    "GPT-OSS 120B (Groq)": ("groq", "openai/gpt-oss-120b"),
    "GPT-OSS 20B (Groq)": ("groq", "openai/gpt-oss-20b"),
    "Qwen3.8 27B (Groq)": ("groq", "qwen/qwen3.8-27b"),
}

_clients = {}


def _gemini():
    if "gemini" not in _clients:
        from google import genai
        from google.genai import types

        _clients["gemini"] = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=60000),
        )
    return _clients["gemini"]


def _groq():
    if "groq" not in _clients:
        from groq import Groq

        _clients["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=60.0)
    return _clients["groq"]


def _clean(text):
    """Remove hidden reasoning blocks some models add."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask(model_name, prompt):
    """Ask one model. Never raises: returns a dict with text or error."""
    kind, model_id = MODELS[model_name]
    start = time.time()
    try:
        if kind == "gemini":
            r = _gemini().models.generate_content(model=model_id, contents=prompt)
            text = r.text
        else:
            r = _groq().chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.choices[0].message.content
        return {"model": model_name, "text": _clean(text), "error": None,
                "seconds": round(time.time() - start, 1)}
    except Exception as e:
        return {"model": model_name, "text": None, "error": str(e)[:200],
                "seconds": round(time.time() - start, 1)}


def ask_two(model_a, model_b, prompt):
    """Ask two models at the same time with the identical prompt."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(ask, model_a, prompt)
        fb = pool.submit(ask, model_b, prompt)
        return fa.result(), fb.result()