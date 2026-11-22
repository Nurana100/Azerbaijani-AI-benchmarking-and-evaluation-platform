import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

QUESTIONS_FILE = "data/benchmark_questions.json"
RESULTS_FILE = "data/results.json"

MODELS = {
    "Gemini (gemini-3.1-flash-lite)": ("gemini", "gemini-3.1-flash-lite"),
    "GPT-OSS 120B (Groq)": ("groq", "openai/gpt-oss-120b"),
    "GPT-OSS 20B (Groq)": ("groq", "openai/gpt-oss-20b"),
    "Qwen3.8 27B (Groq)": ("groq", "qwen/qwen3.8-27b"),
}

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60000),
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=60.0)


def build_prompt(q):
    """Hər modelə eyni formatda sual hazırlayır."""
    parts = []
    if "passage" in q:
        parts.append("Mətn:\n" + q["passage"] + "\n")
    parts.append(q["question"])
    for letter, text in q["choices"].items():
        parts.append(f"{letter}) {text}")
    parts.append("\nYalnız cavab hərfini yaz (A, B, C və ya D). Başqa heç nə yazma.")
    return "\n".join(parts)


def extract_letter(text):
    """Cavabdan ilk A/B/C/D hərfini tapır."""
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None


def ask_model(kind, model_id, prompt):
    if kind == "gemini":
        r = gemini_client.models.generate_content(
            model=model_id, contents=prompt
        )
        return r.text
    r = groq_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def ask_with_retry(kind, model_id, prompt, attempts=4):
    for i in range(attempts):
        try:
            return ask_model(kind, model_id, prompt), None
        except Exception as e:
            error = str(e)[:200]
            print(f"   xəta ({i + 1}/{attempts}): {error}")
            time.sleep(5 * (i + 1))
    return None, error


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    results = load_results()
    done = {(r["question_id"], r["model"]) for r in results}

    for q in questions:
        prompt = build_prompt(q)
        for model_name, (kind, model_id) in MODELS.items():
            if (q["id"], model_name) in done:
                continue
            print(f"{q['id']} -> {model_name}")
            raw, error = ask_with_retry(kind, model_id, prompt)
            letter = extract_letter(raw)
            results.append(
                {
                    "question_id": q["id"],
                    "category": q["category"],
                    "difficulty": q["difficulty"],
                    "model": model_name,
                    "raw_answer": raw,
                    "predicted": letter,
                    "correct_answer": q["correct_answer"],
                    "is_correct": letter == q["correct_answer"],
                    "error": error,
                }
            )
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            time.sleep(1)

    print(f"\nBitdi. {len(results)} nəticə {RESULTS_FILE} faylına yazıldı.")


main()