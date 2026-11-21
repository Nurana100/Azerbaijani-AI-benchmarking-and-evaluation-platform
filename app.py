import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="LexiCode", page_icon="🔤", layout="wide")

# ---------- Project data ----------
# The five evaluation categories. Scores are SAMPLE values only, not real results.
CATEGORIES = [
    {
        "name": "Idioms & Figurative Language",
        "short": "Idioms & Figurative Language",
        "description": "Idioms, phraseological expressions, metaphors and figurative meaning.",
        "sample_score": 72,
    },
    {
        "name": "Orthography",
        "short": "Orthography",
        "description": "Spelling and Azerbaijani-specific letters such as ə, ı, ö, ü, ç, ş, ğ.",
        "sample_score": 85,
    },
    {
        "name": "Morphology & Grammar",
        "short": "Morphology & Grammar",
        "description": "Suffixes, vowel harmony, word structure and grammatical correctness.",
        "sample_score": 68,
    },
    {
        "name": "Context & Comprehension",
        "short": "Context & Comprehension",
        "description": "Reading comprehension and meaning that depends on context.",
        "sample_score": 78,
    },
    {
        "name": "Faithfulness",
        "short": "Faithfulness",
        "description": "Detecting hallucinations and checking that answers stay true to the source.",
        "sample_score": 61,
    },
]

PAGES = ["Overview", "Benchmark", "Model Comparison", "Error Analysis", "AI Arena"]


# ---------- Page functions ----------
def show_overview():
    st.title("LexiCode")
    st.subheader("Azerbaijani Language AI Benchmarking Platform")
    st.write(
        "LexiCode evaluates and compares AI models based on their understanding "
        "of the Azerbaijani language, identifying both their strengths and "
        "weaknesses across linguistic tasks."
    )

    st.warning("All metrics and charts currently display sample data.")

    # Sample metric cards
    df = pd.DataFrame(CATEGORIES)
    average = df["sample_score"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Tested", 2, help="Sample value")
    col2.metric("Benchmark Questions", 30, help="Sample value")
    col3.metric("Categories", len(CATEGORIES))
    col4.metric("Average Score", f"{average:.1f}%", help="Sample value")

    st.divider()

    # Sample chart
    st.subheader("Performance Across Language Categories")
    fig = px.bar(
        df,
        x="short",
        y="sample_score",
        color="sample_score",
        color_continuous_scale="Teal",
        range_y=[0, 100],
        labels={"short": "", "sample_score": "Sample Score (%)"},
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # Category explanations
    st.subheader("What We Evaluate")
    for category in CATEGORIES:
        st.markdown(f"**{category['name']}**: {category['description']}")


def show_placeholder(title, description):
    st.title(title)
    st.info("This page is under construction.")
    st.write(description)


# ---------- Sidebar navigation ----------
st.sidebar.title("AZ LexiCode")
st.sidebar.caption("Azerbaijani AI Benchmark Platform")
page = st.sidebar.radio("Navigation", PAGES)

# ---------- Show the selected page ----------
if page == "Overview":
    show_overview()
elif page == "Benchmark":
    show_placeholder(
        "Benchmark",
        "Browse the human-verified questions by category and difficulty (Easy / Medium / Hard).",
    )
elif page == "Model Comparison":
    show_placeholder(
        "Model Comparison",
        "Compare at least two AI models on identical questions, overall and per category.",
    )
elif page == "Error Analysis":
    show_placeholder(
        "Error Analysis",
        "Real examples of where models succeed and where they fail, with explanations.",
    )
elif page == "AI Arena":
    show_placeholder(
        "AI Arena",
        "Optional demo: type one prompt and see two model answers side by side.",
    )