import json
import os
import random
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="LexiCode", page_icon="🔤", layout="wide")

# ---------- Files ----------
QUESTIONS_FILE = "data/benchmark_questions.json"
RESULTS_FILE = "data/results.json"
ERRORS_FILE = "data/error_analysis.json"
ARENA_VOTES_FILE = "data/arena_votes.json"

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
SHORT = {
    "Idioms & Figurative Language": "Idioms",
    "Orthography": "Orthography",
    "Morphology & Grammar": "Morphology",
    "Context & Comprehension": "Context",
    "Faithfulness": "Faithfulness",
}
LEVELS = ["Easy", "Medium", "Hard"]
PAGES = ["Overview", "Benchmark", "Model Comparison", "Error Analysis", "AI Arena"]

# One fixed color per model, used on every chart (muted flag colors)
MODEL_COLORS = {
    "GPT-OSS 120B (Groq)": "#2563A8",
    "Gemini (gemini-3.1-flash-lite)": "#2F8F5B",
    "GPT-OSS 20B (Groq)": "#B7791F",
    "Qwen3.8 27B (Groq)": "#C8453B",
}
ERROR_COLOR = "#C8453B"
INK = "#1F2937"
GRID = "#E5E7EB"


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


# ---------- Chart helpers ----------
def style(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(size=14, color=INK),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def model_summary(df):
    s = (
        df.groupby("model")
        .agg(correct=("is_correct", "sum"), total=("is_correct", "count"))
        .reset_index()
    )
    s["accuracy"] = s["correct"] / s["total"] * 100
    return s.sort_values(["accuracy", "model"], ascending=[False, True]).reset_index(drop=True)


def leaderboard_chart(summary):
    s = summary.copy()
    s["label"] = s["correct"].astype(int).astype(str) + "/" + s["total"].astype(int).astype(str)
    fig = px.bar(
        s,
        x="accuracy",
        y="model",
        orientation="h",
        text="label",
        color="model",
        color_discrete_map=MODEL_COLORS,
        range_x=[0, 100],
        labels={"accuracy": "Accuracy (%)", "model": ""},
    )
    fig.update_traces(textposition="inside", textfont=dict(size=16, color="#FFFFFF"))
    fig.update_yaxes(categoryorder="array", categoryarray=list(s["model"])[::-1])
    fig.update_layout(showlegend=False)
    return style(fig, 300)


def heatmap_chart(df, summary):
    order = list(summary["model"])
    g = (
        df.groupby(["model", "category"])
        .agg(correct=("is_correct", "sum"), total=("is_correct", "count"))
        .reset_index()
    )
    z, text = [], []
    for m in order:
        z_row, t_row = [], []
        for c in CATEGORY_NAMES:
            row = g[(g["model"] == m) & (g["category"] == c)]
            if row.empty:
                z_row.append(None)
                t_row.append("")
            else:
                cor = int(row["correct"].iloc[0])
                tot = int(row["total"].iloc[0])
                z_row.append(cor / tot * 100)
                t_row.append(f"{cor}/{tot}")
        z.append(z_row)
        text.append(t_row)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[SHORT.get(c, c) for c in CATEGORY_NAMES],
            y=order,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color=INK),
            colorscale=[[0, "#EBB0A8"], [1, "#A9D5B8"]],
            zmin=60,
            zmax=100,
            xgap=3,
            ygap=3,
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style(fig, 320)


def errors_by_category_chart(df):
    errs = (
        df[~df["is_correct"]]
        .groupby("category")
        .size()
        .reindex(CATEGORY_NAMES, fill_value=0)
        .reset_index(name="errors")
    )
    errs["label"] = errs["category"].map(SHORT)
    fig = px.bar(
        errs,
        x="label",
        y="errors",
        text="errors",
        labels={"label": "", "errors": "Wrong answers (all models)"},
    )
    fig.update_traces(marker_color=ERROR_COLOR, textposition="outside")
    fig.update_yaxes(dtick=1, range=[0, max(3, int(errs["errors"].max()) + 1)])
    return style(fig, 300)


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

    questions = load_questions()
    verified = sum(1 for q in questions if q["verified"])
    st.info(
        f"Results come from {len(questions)} questions ({verified} verified by a native speaker). "
        "This is a small sample, so small differences between models are not statistically meaningful."
    )

    summary = model_summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Models tested", df["model"].nunique())
    c2.metric("Questions", len(questions))
    c3.metric("Categories", len(CATEGORIES))
    c4.metric("Average score", f"{df['score'].mean():.1f}%")
    c5.metric("Wrong answers", int((~df["is_correct"]).sum()))

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Model leaderboard")
        st.plotly_chart(leaderboard_chart(summary), width="stretch")
    with right:
        st.subheader("Where the errors fell")
        st.plotly_chart(errors_by_category_chart(df), width="stretch")

    st.subheader("Accuracy by model and category")
    st.caption("Correct answers out of 8 per category. Red cells are where models lost points.")
    st.plotly_chart(heatmap_chart(df, summary), width="stretch")

    with st.expander("What we evaluate"):
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
    st.write(f"All models answered the same {len(load_questions())} questions in the same format.")

    df = results_table()
    if df.empty:
        st.warning("No results yet. Run `python src\\run_benchmark.py` first.")
        return

    summary = model_summary(df)

    st.subheader("Overall results")
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "correct": st.column_config.NumberColumn("Correct", format="%d"),
            "total": st.column_config.NumberColumn("Questions", format="%d"),
            "accuracy": st.column_config.ProgressColumn(
                "Accuracy", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

    st.subheader("Accuracy by category")
    st.caption("Correct answers out of 8 per category. Red cells are where models lost points.")
    st.plotly_chart(heatmap_chart(df, summary), width="stretch")

    st.subheader("Accuracy by difficulty")
    by_level = df.groupby(["model", "difficulty"])["score"].mean().reset_index()
    fig = px.bar(
        by_level,
        x="difficulty",
        y="score",
        color="model",
        barmode="group",
        text_auto=".0f",
        range_y=[0, 110],
        category_orders={"difficulty": LEVELS},
        color_discrete_map=MODEL_COLORS,
        labels={"difficulty": "", "score": "Accuracy (%)", "model": ""},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(style(fig, 380), width="stretch")

    st.caption(
        f"Small sample: {len(load_questions())} questions. "
        "Treat small gaps as noise, and large gaps as a signal worth testing with more questions."
    )


def show_error_analysis():
    st.title("Error Analysis")
    st.write("Real mistakes made by the models, each checked by a native Azerbaijani speaker.")

    errors = load_errors()
    if not errors:
        st.warning("No error analysis file found.")
        return

    questions = {q["id"]: q for q in load_questions()}

    errors_df = pd.DataFrame(errors)
    errors_df["category"] = errors_df["question_id"].map(
        lambda i: SHORT.get(questions[i]["category"], questions[i]["category"]) if i in questions else "Unknown"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Errors by model")
        by_model = errors_df.groupby("model").size().reset_index(name="errors")
        fig = px.bar(
            by_model,
            x="errors",
            y="model",
            orientation="h",
            text="errors",
            color="model",
            color_discrete_map=MODEL_COLORS,
            labels={"errors": "Wrong answers", "model": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(dtick=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 260), width="stretch")
    with right:
        st.subheader("Errors by category")
        by_cat = errors_df.groupby("category").size().reset_index(name="errors")
        fig = px.bar(
            by_cat,
            x="category",
            y="errors",
            text="errors",
            labels={"category": "", "errors": "Wrong answers"},
        )
        fig.update_traces(marker_color=ERROR_COLOR, textposition="outside")
        fig.update_yaxes(dtick=1)
        st.plotly_chart(style(fig, 260), width="stretch")

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


def save_vote(entry):
    votes = read_json(ARENA_VOTES_FILE)
    votes.append(entry)
    with open(ARENA_VOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(votes, f, ensure_ascii=False, indent=2)


def show_arena():
    from src.llm import MODELS, ask_two

    st.title("AI Arena")
    st.write("Type a question in Azerbaijani. Two models answer, shown side by side without names.")
    st.info(
        "Arena votes are personal opinions on free-form questions. "
        "They are NOT part of the benchmark scores, because these questions have no answer key."
    )

    names = list(MODELS)
    col1, col2 = st.columns(2)
    model_1 = col1.selectbox("Model 1", names, index=0)
    model_2 = col2.selectbox("Model 2", names, index=1)
    prompt = st.text_area("Your question", height=100)

    if st.button("Compare models", type="primary"):
        if model_1 == model_2:
            st.warning("Choose two different models.")
        elif not prompt.strip():
            st.warning("Type a question first.")
        else:
            pair = [model_1, model_2]
            random.shuffle(pair)  # blind: random order every time
            with st.spinner("Asking both models..."):
                result_a, result_b = ask_two(pair[0], pair[1], prompt.strip())
            st.session_state["arena"] = {
                "prompt": prompt.strip(), "a": result_a, "b": result_b, "voted": None,
            }

    state = st.session_state.get("arena")
    if state:
        st.divider()
        st.markdown(f"**Question:** {state['prompt']}")
        left, right = st.columns(2)
        for column, label, result in ((left, "A", state["a"]), (right, "B", state["b"])):
            with column:
                st.subheader(f"Answer {label}")
                if result["error"]:
                    st.error(f"This model failed: {result['error']}")
                else:
                    st.write(result["text"])

        both_ok = not state["a"]["error"] and not state["b"]["error"]
        if both_ok and not state["voted"]:
            st.write("Which answer is better?")
            b1, b2, b3, b4 = st.columns(4)
            choice = None
            if b1.button("A is better", use_container_width=True):
                choice = "A"
            if b2.button("B is better", use_container_width=True):
                choice = "B"
            if b3.button("Tie", use_container_width=True):
                choice = "Tie"
            if b4.button("Both bad", use_container_width=True):
                choice = "Both bad"
            if choice:
                save_vote({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "prompt": state["prompt"],
                    "model_a": state["a"]["model"],
                    "model_b": state["b"]["model"],
                    "answer_a": state["a"]["text"],
                    "answer_b": state["b"]["text"],
                    "winner": choice,
                })
                state["voted"] = choice

        if state["voted"]:
            st.success(f"Vote saved: {state['voted']}")
            st.write(f"**A** = {state['a']['model']} ({state['a']['seconds']}s)")
            st.write(f"**B** = {state['b']['model']} ({state['b']['seconds']}s)")

    votes = read_json(ARENA_VOTES_FILE)
    if votes:
        with st.expander(f"Vote summary ({len(votes)} votes)"):
            wins = {}
            for v in votes:
                for m in (v["model_a"], v["model_b"]):
                    wins.setdefault(m, {"wins": 0, "matches": 0})["matches"] += 1
                if v["winner"] in ("A", "B"):
                    winner = v["model_a"] if v["winner"] == "A" else v["model_b"]
                    wins[winner]["wins"] += 1
            st.dataframe(
                pd.DataFrame(wins).T.reset_index().rename(columns={"index": "model"}),
                hide_index=True,
            )


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
    show_arena()