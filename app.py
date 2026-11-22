import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LexiCode", page_icon="🔤", layout="wide")

# ---------- Files ----------
QUESTIONS_FILE = "data/benchmark_questions.json"
RESULTS_FILE = "data/results.json"
ERRORS_FILE = "data/error_analysis.json"

# ---------- Project constants ----------
CATEGORIES = [
    {
        "name": "Idioms & Figurative Language",
        "description": "Idioms, phraseological expressions, metaphors and figurative meaning.",
    },
    {
        "name": "Orthography",
        "description": "Spelling and Azerbaijani-specific letters such as ə, ı, ö, ü, ç, ş, ğ.",
    },
    {
        "name": "Morphology & Grammar",
        "description": "Suffixes, vowel harmony, word structure and grammatical correctness.",
    },
    {
        "name": "Context & Comprehension",
        "description": "Reading comprehension and meaning that depends on context.",
    },
    {
        "name": "Faithfulness",
        "description": "Detecting hallucinations and checking that answers stay true to the source.",
    },
]
CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
LEVELS = ["Easy", "Medium", "Hard"]
PAGES = ["Overview", "Benchmark", "Model Comparison", "Error Analysis", "AI Arena"]


# ---------- Data loading ----------
def read_json(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_questions():
    return read_json(QUESTIONS_FILE)


@st.cache_data
def load_results():
    return read_json(RESULTS_FILE)


@st.cache_data
def load_errors():
    return read_json(ERRORS_FILE)


def results_table():
    df = pd.DataFrame(load_results())
    if not df.empty:
        df["score"] = df["is_correct"].astype(int) * 100
    return df


# ---------- Pages ----------
def show_overview():
    st.title("LexiCode")
    st.subheader("Azerbaijani Language AI Benchmarking Platform")
    st.write(
        "LexiCode evaluates and compares AI models based on their understanding "
        "of the Azerbaijani language, identifying both their strengths and "
        "weaknesses across linguistic tasks."
    )

    df = results_table()
    if df.empty:
        st.warning("No results yet. Run `python src\\run_benchmark.py` first.")
        return

    st.info(
        "Results come from 30 human-verified questions. This is a small sample, "
        "so small differences between models are not statistically meaningful."
    )

    questions = load_questions()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Tested", df["model"].nunique())
    col2.metric("Benchmark Questions", len(questions))
    col3.metric("Categories", len(CATEGORIES))
    col4.metric("Average Score", f"{df['score'].mean():.1f}%")

    st.divider()

    st.subheader("Performance Across Language Categories")
    st.caption("Average accuracy of all tested models in each category.")
    by_category = df.groupby("category")["score"].mean().reset_index()
    fig = px.bar(
        by_category,
        x="category",
        y="score",
        color="score",
        color_continuous_scale="Teal",
        range_y=[0, 100],
        category_orders={"category": CATEGORY_NAMES},
        labels={"category": "", "score": "Accuracy (%)"},
    )
    st.plotly_chart(fig, width="stretch")

    # Strengths and weaknesses are calculated from the data
    best = by_category[by_category["score"] == by_category["score"].max()]
    worst = by_category[by_category["score"] == by_category["score"].min()]
    st.subheader("Strengths and Weaknesses")
    st.success("Strongest: " + ", ".join(best["category"]) + f" ({best['score'].iloc[0]:.1f}%)")
    st.error("Weakest: " + ", ".join(worst["category"]) + f" ({worst['score'].iloc[0]:.1f}%)")

    st.divider()

    st.subheader("What We Evaluate")
    for category in CATEGORIES:
        st.markdown(f"**{category['name']}**: {category['description']}")


def show_benchmark():
    st.title("Benchmark")
    st.write("Browse the benchmark questions by category and difficulty.")

    questions = load_questions()

    col1, col2 = st.columns(2)
    chosen_categories = col1.multiselect("Category", CATEGORY_NAMES, default=CATEGORY_NAMES)
    chosen_levels = col2.multiselect("Difficulty", LEVELS, default=LEVELS)

    filtered = [
        q for q in questions
        if q["category"] in chosen_categories and q["difficulty"] in chosen_levels
    ]

    verified_count = sum(1 for q in filtered if q["verified"])
    st.caption(f"Showing {len(filtered)} questions. Human-verified: {verified_count}.")

    for q in filtered:
        status = "Verified" if q["verified"] else "Not yet verified"
        with st.expander(f"{q['id']}  |  {q['category']}  |  {q['difficulty']}  |  {status}"):
            if "passage" in q:
                st.info(q["passage"])
            st.markdown(f"**{q['question']}**")
            for letter, text in q["choices"].items():
                st.write(f"{letter}) {text}")
            st.success(f"Correct answer: {q['correct_answer']}")
            st.caption(q["explanation"])


def show_model_comparison():
    st.title("Model Comparison")
    st.write("All models answered the same 30 questions in the same format.")

    df = results_table()
    if df.empty:
        st.warning("No results yet. Run `python src\\run_benchmark.py` first.")
        return

    # Overall table
    summary = (
        df.groupby("model")
        .agg(correct=("is_correct", "sum"), total=("is_correct", "count"), accuracy=("score", "mean"))
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary["accuracy"] = summary["accuracy"].round(1)
    st.subheader("Overall Results")
    st.dataframe(summary, hide_index=True, width="stretch")

    # By category
    st.subheader("Accuracy by Category")
    by_model_category = df.groupby(["model", "category"])["score"].mean().reset_index()
    fig = px.bar(
        by_model_category,
        x="category",
        y="score",
        color="model",
        barmode="group",
        range_y=[0, 100],
        category_orders={"category": CATEGORY_NAMES},
        labels={"category": "", "score": "Accuracy (%)", "model": "Model"},
    )
    st.plotly_chart(fig, width="stretch")

    # By difficulty
    st.subheader("Accuracy by Difficulty")
    by_level = df.pivot_table(index="model", columns="difficulty", values="score", aggfunc="mean")
    by_level = by_level.reindex(columns=[l for l in LEVELS if l in by_level.columns]).round(1)
    st.dataframe(by_level, width="stretch")

    st.caption(
        "Only 30 questions (6 per category). Treat small gaps as noise, "
        "and large gaps as a signal worth testing with more questions."
    )


def show_error_analysis():
    st.title("Error Analysis")
    st.write("Real mistakes made by the models, each checked by a native Azerbaijani speaker.")

    errors = load_errors()
    if not errors:
        st.warning("No error analysis file found.")
        return

    questions = {q["id"]: q for q in load_questions()}

    models = sorted({e["model"] for e in errors})
    chosen = st.multiselect("Model", models, default=models)
    shown = [e for e in errors if e["model"] in chosen]
    st.caption(f"Showing {len(shown)} of {len(errors)} errors.")

    for e in shown:
        q = questions.get(e["question_id"])
        title = f"{e['question_id']}  |  {e['model']}  |  {e['error_type']}"
        with st.expander(title):
            if q:
                if "passage" in q:
                    st.info(q["passage"])
                st.markdown(f"**{q['question']}**")
                for letter, text in q["choices"].items():
                    st.write(f"{letter}) {text}")
            col1, col2 = st.columns(2)
            col1.error(f"Model answered: {e['predicted']}")
            col2.success(f"Correct answer: {e['correct_answer']}")
            st.write(e["note"])


def show_placeholder(title, description):
    st.title(title)
    st.info("This page is under construction.")
    st.write(description)


# ---------- Sidebar ----------
st.sidebar.title("AZ LexiCode")
st.sidebar.caption("Azerbaijani AI Benchmark Platform")
page = st.sidebar.radio("Navigation", PAGES)

# ---------- Show the selected page ----------
if page == "Overview":
    show_overview()
elif page == "Benchmark":
    show_benchmark()
elif page == "Model Comparison":
    show_model_comparison()
elif page == "Error Analysis":
    show_error_analysis()
elif page == "AI Arena":
    show_placeholder(
        "AI Arena",
        "Optional demo: type one prompt and see two model answers side by side.",
    )