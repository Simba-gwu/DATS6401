"""Pure analysis functions for the DATS6401 Week 7 news visualization app.

This module deliberately has no Streamlit dependency so its methods can be
unit-tested and reused in notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{2,}\b"


def normalize_articles(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize BBC `news,type` and common `text,category` CSV schemas."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    text_name = next((x for x in ("news", "text", "article", "content") if x in df), None)
    category_name = next((x for x in ("type", "category", "label") if x in df), None)
    if not text_name or not category_name:
        raise ValueError("CSV needs 'news' and 'type' columns (or 'text' and 'category').")
    df = df[[text_name, category_name]].rename(
        columns={text_name: "text", category_name: "category"}
    )
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df = df[(df["text"].str.len() >= 50) & (df["category"].str.len() > 0)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable articles found (each text must contain at least 50 characters).")
    return df


def analyze_texts(texts: tuple[str, ...], n_topics: int) -> dict:
    """Compute transparent word statistics and unsupervised NMF topics."""
    if len(texts) < max(20, n_topics):
        raise ValueError("Select at least 20 articles to analyze topics reliably.")

    # Count ALL retained word types: no vocabulary cutoff for honest frequencies.
    counter = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_PATTERN,
        stop_words="english",
    )
    count_matrix = counter.fit_transform(texts)
    terms = counter.get_feature_names_out()
    total_occurrences = np.asarray(count_matrix.sum(axis=0)).ravel()
    document_frequencies = np.asarray((count_matrix > 0).sum(axis=0)).ravel()
    frequencies = pd.DataFrame({
        "word": terms,
        "occurrences": total_occurrences.astype(int),
        "documents": document_frequencies.astype(int),
    })
    frequencies["document_pct"] = 100.0 * frequencies["documents"] / len(texts)

    # Separate, pruned TF-IDF representation for topic modeling.
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_PATTERN,
        stop_words="english",
        min_df=2,
        max_df=0.90,
        max_features=1800,
        sublinear_tf=True,
    )
    tfidf = tfidf_vectorizer.fit_transform(texts)
    if tfidf.shape[1] < n_topics:
        raise ValueError("Too few distinct words to fit the selected number of topics.")

    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=350)
    weights = nmf.fit_transform(tfidf)
    names = tfidf_vectorizer.get_feature_names_out()
    topic_keywords = []
    topic_rows = []
    for idx, topic_component in enumerate(nmf.components_):
        order = np.argsort(topic_component)[::-1][:12]
        keywords = names[order].tolist()
        topic_keywords.append(keywords)
        for word_idx in order:
            topic_rows.append({
                "topic": idx + 1,
                "word": str(names[word_idx]),
                "weight": float(topic_component[word_idx]),
            })

    valid = np.asarray(tfidf.sum(axis=1)).ravel() > 0
    assigned = np.where(valid, weights.argmax(axis=1) + 1, 0)
    topic_counts = pd.Series(assigned[assigned > 0]).value_counts().sort_index()

    mean_tfidf = np.asarray(tfidf.mean(axis=0)).ravel()
    tfidf_means = pd.DataFrame({"word": names, "mean_tfidf": mean_tfidf})

    return {
        "frequencies": frequencies,
        "mean_tfidf": tfidf_means,
        "topic_terms": pd.DataFrame(topic_rows),
        "keywords": topic_keywords,
        "assigned_topics": assigned,
        "topic_weights": weights,
        "topic_counts": topic_counts,
        "tfidf": tfidf,
        "n_terms": len(frequencies),
        "retained_occurrences": int(total_occurrences.sum()),
        "excluded_docs": int((~valid).sum()),
    }


def project_documents(tfidf) -> tuple[np.ndarray, float]:
    """Optional lightweight *lexical* 2D projection; not a neural embedding."""
    if min(tfidf.shape) < 3:
        raise ValueError("Need at least three documents and terms for projection.")
    reducer = TruncatedSVD(n_components=2, random_state=42)
    coords = reducer.fit_transform(tfidf)
    return coords, float(reducer.explained_variance_ratio_.sum())
