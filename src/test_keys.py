import os

from dotenv import load_dotenv

load_dotenv()

QUESTION = (
    "Hansı sözün yazılışı düzgündür?\n"
    "A) qardas\nB) qardaş\nC) qardash\nD) qardaç\n\n"
    "Yalnız cavab hərfini yaz (A, B, C və ya D)."
)

# ---------- Gemini ----------
print("Gemini yoxlanılır...")
try:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=30000),  # 30 saniyə (millisaniyə ilə)
    )
    response = client.models.generate_content(
         model="gemini-3.1-flash-lite", contents=QUESTION
    )
    print("Gemini cavabı:", response.text.strip())
except Exception as e:
    print("Gemini XƏTASI:", e)

# ---------- Groq ----------
print("Groq yoxlanılır...")
try:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=30.0)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": QUESTION}],
    )
    print("Groq cavabı:", response.choices[0].message.content.strip())
except Exception as e:
    print("Groq XƏTASI:", e)