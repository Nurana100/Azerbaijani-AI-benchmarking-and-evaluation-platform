\# LexiCode: Azerbaijani AI Benchmarking \& Evaluation Platform



LexiCode evaluates and compares \*\*existing\*\* AI models on Azerbaijani-language tasks, showing both strengths and weaknesses. It is a benchmarking project, not a model-training project.



\## Categories (6 questions each, 30 total)

1\. Idioms \& Figurative Language

2\. Orthography

3\. Morphology \& Grammar

4\. Context \& Comprehension

5\. Faithfulness (hallucination detection)



All questions are multiple choice and were verified by a native Azerbaijani speaker. Each has a difficulty level: Easy, Medium or Hard.



\## Method

Every model receives the same prompt for each question and must answer with a single letter. Answers are scored automatically against the verified key. Wrong answers were manually reviewed to confirm they are model errors, not question flaws.



\## Models tested

Gemini (gemini-3.1-flash-lite), GPT-OSS 120B, GPT-OSS 20B, Qwen3.8 27B (the last three via Groq).



\## Key findings

\- The two strongest models scored 30/30, so this question set cannot separate them (ceiling effect).

\- Smaller models made 2-3 mistakes, mainly on Azerbaijani-specific letters (ə), vowel harmony, suffix order and idioms.

\- No model made errors in Context \& Comprehension or Faithfulness.



\## Limitations

\- Only 30 questions: small differences are not statistically meaningful.

\- Multiple choice only; free-text answers are future work.

\- Audio evaluation is future scope.



\## Run locally

&#x20;   pip install -r requirements.txt

&#x20;   streamlit run app.py



To re-run the benchmark, add `GEMINI\_API\_KEY` and `GROQ\_API\_KEY` to a `.env` file, then run `python src\\run\_benchmark.py`.



\## Next steps

Harder questions, more models, free-text scoring, audio evaluation.

