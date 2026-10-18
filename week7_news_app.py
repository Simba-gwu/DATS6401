"""DATS6401 Week 7: honest frequency, NMF topic modeling and optional SVD view."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

from week7_news_analysis import normalize_articles, analyze_texts, project_documents

ROOT = Path(__file__).resolve().parent
LOCAL_CSV = ROOT / "data" / "week7_bbc_news.csv"
REMOTE_CSV = (
    "https://raw.githubusercontent.com/suraj-deshmukh/"
    "BBC-Dataset-News-Classification/master/dataset/dataset.csv"
)

st.set_page_config(page_title="News Language Explorer | Week 7", layout="wide")
st.title("News Language Explorer")
st.caption("DATS6401 · Week 7 · historical BBC news corpus (2004–2005)")
st.write(
    "Explore which terms occur most often, which themes a topic model discovers, "
    "and how articles relate in a lightweight two-dimensional text projection. "
    "**This is a historical corpus, not today's news.**"
)

@st.cache_data(show_spinner=False)
def read_local(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns  # ensures cache changes when local file changes
    return normalize_articles(pd.read_csv(path, encoding_errors="replace"))

@st.cache_data(show_spinner="Fetching the BBC news dataset...")
def read_remote(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return normalize_articles(pd.read_csv(BytesIO(response.content), encoding_errors="replace"))

@st.cache_data(show_spinner="Fitting the NMF topic model...")
def cached_analysis(texts: tuple[str, ...], topics: int) -> dict:
    return analyze_texts(texts, topics)

@st.cache_data(show_spinner="Computing the 2D SVD projection...")
def cached_projection(_tfidf, texts: tuple[str, ...], topics: int):
    # _tfidf is not hashable, so texts + topics act as the cache key
    return project_documents(_tfidf)

st.sidebar.header("Explore the corpus")
uploaded = st.sidebar.file_uploader(
    "Optional: upload your own news CSV",
    type="csv",
    help="Use columns news/type or text/category. Uploaded data overrides the BBC source.",
)

try:
    if uploaded is not None:
        articles = normalize_articles(pd.read_csv(uploaded, encoding_errors="replace"))
        source_label = "Your uploaded CSV"
    elif LOCAL_CSV.exists():
        articles = read_local(str(LOCAL_CSV), LOCAL_CSV.stat().st_mtime_ns)
        source_label = "BBC news CSV (local data file)"
    else:
        articles = read_remote(REMOTE_CSV)
        source_label = "BBC news CSV (downloaded from public repository)"
except Exception as exc:
    st.error("Could not load the news dataset.")
    st.write(
        "Run `python download_week7_news.py` in the project folder, or upload the CSV "
        "using the sidebar. The expected columns are `news,type` or `text,category`."
    )
    st.code(REMOTE_CSV)
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

available_categories = sorted(articles["category"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "News categories", available_categories, default=available_categories
)
if not selected_categories:
    st.warning("Choose at least one category in the sidebar.")
    st.stop()

available = articles.loc[articles["category"].isin(selected_categories)].copy()
if len(available) < 20:
    st.warning("Select categories with at least 20 documents.")
    st.stop()

max_options = sorted({min(n, len(available)) for n in (200, 400, 600, 800, 1000, 1200, 1500, 1800, 2200, len(available))})
chosen_max = st.sidebar.select_slider(
    "Maximum articles (sampled with seed 42)", options=max_options,
    value=min(1200, len(available)),
)
n_topics = st.sidebar.slider("Number of NMF topics", 3, 8, 5)
frequency_view = st.sidebar.radio(
    "Frequency metric",
    ["Total word occurrences", "Documents containing the word", "Mean TF-IDF weight"],
    help="The denominator and interpretation change with this choice.",
)
top_n = st.sidebar.slider("Words to display", 10, 30, 15, 5)
show_projection = st.sidebar.checkbox(
    "Compute optional 2D projection", value=False,
    help="TruncatedSVD on TF-IDF. This is a lexical projection, NOT a neural embedding.",
)

if len(available) > chosen_max:
    subset = available.sample(n=chosen_max, random_state=42).reset_index(drop=True)
else:
    subset = available.reset_index(drop=True)

texts = tuple(subset["text"].tolist())
try:
    analysis = cached_analysis(texts, n_topics)
except (ValueError, RuntimeError) as exc:
    st.error(f"Unable to analyze this selection: {exc}")
    st.stop()

subset["topic_id"] = analysis["assigned_topics"]
subset["topic"] = subset["topic_id"].map(
    lambda n: f"Topic {n}" if n > 0 else "Unassigned"
)

st.caption(f"**Source:** {source_label} · {len(subset):,} analyzed articles of {len(articles):,} usable articles")
a, b, c, d = st.columns(4)
a.metric("Documents analyzed", f"{len(subset):,}")
b.metric("Categories", len(selected_categories))
c.metric("Distinct retained words", f"{analysis['n_terms']:,}")
d.metric("NMF topics", n_topics)

with st.expander("How to use this dashboard", expanded=False):
    st.markdown(
        "Choose categories and sample size in the sidebar. In **Word frequency**, change "
        "between total occurrences, document frequency, and average TF-IDF. "
        "In **Topics**, select one discovered topic to inspect keywords and associated "
        "articles. Enable the optional projection in the sidebar to display the fourth view. "
        "These are **unsupervised** model topics; they do not necessarily correspond to BBC's categories."
    )

tab_overview, tab_frequency, tab_topics, tab_projection, tab_notes = st.tabs(
    ["Dataset", "Word frequency", "NMF topics", "2D projection (optional)", "Interpretation & limitations"]
)

with tab_overview:
    st.subheader("What does this corpus represent?")
    st.write(
        "Each row is a BBC news article (2004–2005), with article text and an existing "
        "editorial category: Business, Entertainment, Politics, Sport, or Tech. "
        "The BBC categories are used here only for exploration and comparison; "
        "the NMF topic model is not trained on those labels."
    )
    counts = subset["category"].value_counts().rename_axis("category").reset_index(name="articles")
    fig_category = px.bar(counts, x="category", y="articles", text="articles", title="Documents in the analyzed sample")
    fig_category.update_yaxes(rangemode="tozero", title="Articles")
    st.plotly_chart(fig_category, width="stretch")
    st.write("Article excerpts (not entire articles) from your current selection:")
    preview = subset[["category", "topic", "text"]].copy()
    preview["text"] = preview["text"].str.slice(0, 210) + "…"
    st.dataframe(preview.head(25), width="stretch", hide_index=True)
    st.download_button(
        "Download analyzed articles (CSV)",
        data=subset[["category", "text", "topic"]].to_csv(index=False).encode("utf-8"),
        file_name="week7_selected_news.csv", mime="text/csv",
        help="Download for personal analysis; review the source dataset's rights before republishing articles.",
    )

with tab_frequency:
    st.subheader("Word frequency — what is common?")
    st.write(
        "We lowercase words, keep alphabetic tokens of 2+ letters, and remove "
        "scikit-learn's English stop-word list. No vocabulary cutoff is used for "
        "the raw count and document-frequency charts."
    )
    if frequency_view == "Total word occurrences":
        frame = analysis["frequencies"].nlargest(top_n, "occurrences")
        x_field = "occurrences"
        x_label = "Total times the word appears in the sampled articles"
        foot = "A word repeated 20 times in one article contributes 20 occurrences."
    elif frequency_view == "Documents containing the word":
        frame = analysis["frequencies"].nlargest(top_n, "documents")
        x_field = "documents"
        x_label = "Number of sampled articles containing the word at least once"
        foot = f"Document frequency counts an article only once per word (out of {len(subset):,} articles)."
    else:
        frame = analysis["mean_tfidf"].nlargest(top_n, "mean_tfidf")
        x_field = "mean_tfidf"
        x_label = "Mean normalized TF-IDF weight across sampled articles"
        foot = "Mean TF-IDF measures lexical salience, not raw frequency or category-specific keyness."
    frame = frame.sort_values(x_field)
    fig_words = px.bar(frame, x=x_field, y="word", orientation="h", title=frequency_view,
                       labels={x_field: x_label, "word": "Word"})
    fig_words.update_xaxes(rangemode="tozero")
    fig_words.update_layout(height=max(450, top_n * 25), yaxis_title="Word")
    st.plotly_chart(fig_words, width="stretch")
    st.caption(foot)
    st.info(
        "**Reveals:** dominant terms across the selected articles. **Can mislead:** long articles "
        "can dominate raw counts, common words need not indicate topics, and preprocessing "
        "choices affect rankings. The horizontal bar axis starts at zero."
    )

with tab_topics:
    st.subheader("NMF topic modeling — what themes recur?")
    st.write(
        "NMF decomposes a TF-IDF document–word matrix into latent word groups. "
        "The model is fitted to your selected articles only. Topic numbers are "
        "arbitrary labels, not editorial subject categories."
    )
    topic_summary = pd.DataFrame({
        "topic": [f"Topic {t+1}" for t in range(n_topics)],
        "keywords": [", ".join(k[:6]) for k in analysis["keywords"]],
        "assigned_articles": [int(analysis["topic_counts"].get(t+1, 0)) for t in range(n_topics)],
    })
    st.dataframe(topic_summary, width="stretch", hide_index=True)
    topic_num = st.selectbox("Select a topic to explore", range(1, n_topics + 1),
                             format_func=lambda idx: f"Topic {idx}: {', '.join(analysis['keywords'][idx-1][:3])}")
    topic_terms = analysis["topic_terms"].query("topic == @topic_num").sort_values("weight")
    fig_topic = px.bar(topic_terms, x="weight", y="word", orientation="h",
                       title=f"Topic {topic_num}: top NMF keywords",
                       labels={"weight": "NMF term weight (relative, not article count)", "word": "Keyword"})
    fig_topic.update_xaxes(rangemode="tozero")
    st.plotly_chart(fig_topic, width="stretch")
    st.write("Articles most strongly associated with this topic (model-based):")
    scores = analysis["topic_weights"][:, topic_num - 1]
    ranked = np.argsort(scores)[::-1][:5]
    examples = subset.iloc[ranked][["category", "text"]].copy()
    examples["topic_strength"] = scores[ranked].round(3)
    examples["excerpt"] = examples["text"].str.slice(0, 260) + "…"
    st.dataframe(examples[["category", "topic_strength", "excerpt"]],
                 width="stretch", hide_index=True)
    st.info(
        "**Reveals:** recurring themes and words that characterize each topic. "
        "**Can mislead:** NMF topics are mathematical mixtures; the labels and number "
        "of topics are choices, and one article may discuss several themes. "
        "The bar values are relative model weights, not word counts or probabilities."
    )

with tab_projection:
    st.subheader("Optional TF-IDF / TruncatedSVD projection")
    st.write(
        "This plot projects articles from TF-IDF space into two coordinates using "
        "TruncatedSVD (a form of latent semantic analysis). **It is not a neural embedding "
        "such as BERTopic's sentence embeddings.**"
    )
    if show_projection:
        coords, explained = cached_projection(analysis["tfidf"], texts, n_topics)
        scatter_data = pd.DataFrame({
            "SVD component 1": coords[:, 0],
            "SVD component 2": coords[:, 1],
            "Detected topic": subset["topic"],
            "BBC category": subset["category"],
            "excerpt": subset["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 120),
        })
        color_by = st.radio("Color points by", ["Detected topic", "BBC category"], horizontal=True)
        fig_scatter = px.scatter(scatter_data, x="SVD component 1", y="SVD component 2",
                                 color=color_by, hover_data=["excerpt"], opacity=0.67,
                                 title="Articles in a 2D lexical projection")
        st.plotly_chart(fig_scatter, width="stretch")
        st.caption(f"The two components explain approximately {explained:.1%} of TF-IDF feature variance.")
    else:
        st.info("Turn on **Compute optional 2D projection** in the sidebar to see this view.")
    st.warning(
        "**Limitation:** 2D distances can exaggerate or conceal similarity in the original "
        "high-dimensional text. A cluster does not prove a real social or editorial grouping."
    )

with tab_notes:
    st.subheader("Interpretation and limitations")
    st.markdown(
        "**Frequency view:** The most common retained words reveal shared vocabulary. "
        "However, total counts reward repetition in long articles; document frequency "
        "offers a different perspective. Stop-word filtering also changes what is visible.\n\n"
        "**NMF topic view:** Topic keywords summarize co-occurring terms and help us "
        "inspect themes. But selecting five or six topics does not mean the corpus has "
        "exactly that many natural themes; an article can contain more than one.\n\n"
        "**Optional projection:** A two-dimensional approximation can reveal broad "
        "lexical neighborhoods. It compresses a large vocabulary into only two axes, "
        "so apparent distances are not reliable measurements of semantic similarity.\n\n"
        "**Sampling and source limitations:** This BBC news collection is historical "
        "(2004–2005), limited to five editorial categories, and reflects one publisher's "
        "coverage. Your sidebar category choices and seed-42 sampling affect what is shown. "
        "The corpus cannot represent all news or current public opinion."
    )
    with st.expander("Data and methods"):
        st.markdown(
            "- Source: [BBC news classification dataset (community mirror)]("
            "https://github.com/suraj-deshmukh/BBC-Dataset-News-Classification).\n"
            "- Rows are article texts, not web-scraped live results.\n"
            "- Frequency: `CountVectorizer` with English stop words and alphabetic 2+ letter tokens.\n"
            "- Topic modeling: TF-IDF (max 1,800 terms, min document frequency 2, "
            "max document fraction 90%) followed by NMF with fixed seed 42.\n"
            "- Optional projection: `TruncatedSVD` on the same TF-IDF document matrix.\n"
            "- Caching: Streamlit caches the data load, NMF analysis and optional projection."
        )

st.divider()
st.caption("Week 7 · Reproducible, exploratory text visualization · Data: historical BBC news articles")
