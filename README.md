# LexiCode

LexiCode is a small Python and Streamlit platform that evaluates and compares **existing** AI models on Azerbaijani-language tasks. It shows both their strengths and their weaknesses. It is a benchmarking project, not a model-training project.

## The question

AI models write fluent Azerbaijani, but fluent does not mean correct. LexiCode asks: how well do AI models really understand Azerbaijani, and how can we measure it?

## The benchmark

The benchmark has **40 multiple-choice questions**, 8 in each of 5 categories: Idioms & Figurative Language, Orthography, Morphology & Grammar, Context & Comprehension, and Faithfulness. Each question is marked Easy, Medium or Hard and was verified by a native Azerbaijani speaker.

## Method

Every model receives the same prompt for each question and must answer with a single letter. Answers are scored automatically against the verified key. Every wrong answer was reviewed by hand to confirm it was a model error and not a question flaw. One question (CTX-007) was reworded after its first wording turned out to be ambiguous, and was then re-run on all models.

## Results

| Model | Access | Score |
|---|---|---|
| GPT-OSS 120B | Groq | 40/40 |
| Gemini (gemini-3.1-flash-lite) | Gemini API | 39/40 |
| GPT-OSS 20B | Groq | 38/40 |
| Qwen3.8 27B | Groq | 36/40 |

The top models are separated by a single question, so this question set cannot reliably rank them. The smaller models made mistakes on Azerbaijani-specific letters, vowel harmony, suffix order and idioms. Two errors fell in Context & Comprehension, both on multi-step reasoning.

## The platform

The dashboard has five pages: **Overview**, **Benchmark**, **Model Comparison**, **Error Analysis** and **AI Arena**. In the Arena you type any question and two models answer side by side, with their names hidden until you vote. Arena votes are opinions on free-form questions and are **not** part of the benchmark scores.

## Limitations

With only 40 questions, small differences between models are not statistically meaningful. All questions are multiple choice, so free-text answers are not yet scored. Audio and pronunciation are not covered.

## Run locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

To re-run the benchmark, create a `.env` file with `GEMINI_API_KEY` and `GROQ_API_KEY`, then run `python src\run_benchmark.py`.

## Next steps

More and harder questions, more models, free-text scoring and audio evaluation.