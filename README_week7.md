# Week 7 — News Language Explorer (DATS6401)

## What the app does

Analyzes historical BBC news articles (2004–2005; Business, Entertainment,
Politics, Sport, Tech). It includes:

- **Honest word-frequency bar charts**: total retained occurrences vs document
  frequency vs corpus-average TF-IDF, zero-baseline axes and metric definitions.
- **NMF topic-model explorer**: selected topic's keywords, topic counts and
  model-ranked example articles.
- **Optional lightweight TF-IDF/TruncatedSVD 2D projection**: comparison of
  model topics and BBC article categories. This is **not neural BERTopic embeddings**.
- Sidebar widgets for category, sample size, topic count, frequency metric,
  number of words, and optional projection.
- Dataset/interpretation tabs, limitations, caching and CSV export.

## Install & run on your Mac

Open the existing project folder in VS Code (the folder name starts with a space):

```bash
cd "/Users/simba/Documents/ DATS6401"
conda activate dataviz
python -m pip install -r requirements.txt
python download_week7_news.py
python -m streamlit run week7_news_app.py
```

The app also attempts to fetch the dataset automatically from the public
repository if `data/week7_bbc_news.csv` is missing. If that fails, run the
script or upload a CSV from the app sidebar.

## Data acquisition & columns

Source: https://github.com/suraj-deshmukh/BBC-Dataset-News-Classification

Direct CSV: https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv

The downloaded CSV contains two columns: `news` (article text) and `type`
(editorial category). The local script saves to `data/week7_bbc_news.csv`.
The app also accepts an uploaded CSV with `news,type` or `text,category`.

The dataset is a historical teaching corpus, **not live news**. Check source
rights before republishing article text. For the course submission you can
commit this download script rather than redistributing the full BBC corpus.

## Integrate into your existing GitHub repo

Place these files **at the root** of `Simba-gwu/DATS6401`:

- `week7_news_app.py`
- `week7_news_analysis.py`
- `download_week7_news.py`
- `requirements.txt` (keep one clean requirements file for all course apps)
- `README_week7.md` (optional but recommended)

You do **not** have to commit the downloaded BBC CSV: Streamlit Cloud can
fetch it when the app starts; this does require external connectivity. The
submitted download script documents how to obtain the data. If your Cloud
instance cannot fetch it, use the sidebar uploader during testing or check
whether the dataset's redistribution terms permit you to include it in your
repo.

```bash
git add week7_news_app.py week7_news_analysis.py download_week7_news.py requirements.txt README_week7.md
git commit -m "Add Week 7 news text visualization app"
git push
```

On https://share.streamlit.io create a **new app** under repo
`Simba-gwu/DATS6401`, branch `main`, and main-file path `week7_news_app.py`.
Wait for deployment, open the live link, and test all tabs/widgets.

**Blackboard submission:** paste the *actual live Streamlit URL* and the
GitHub repository link. Provide the download script or dataset file as requested.
Do not invent a URL before Streamlit Cloud assigns it.

## Methods and honest interpretation

- Word frequency excludes English stopwords; totals count repeated mentions,
  while document frequency counts each article at most once per word.
- NMF uses TF-IDF and finds latent topics. Top terms are **model weights**, not
  probabilities or document-counts, and do not necessarily match the BBC labels.
- A 2D TruncatedSVD visualization is a **lexical vector projection**, not an
  LLM/neural embedding. Distances are distorted by dimensionality reduction.
- Sample is deterministic (`random_state=42`); changing categories, size, or
  number of topics can change the patterns. BBC coverage is from 2004–2005.

## Trouble-shooting

- `ModuleNotFoundError: No module named 'sklearn'`: install **scikit-learn**
  using `python -m pip install scikit-learn`, then run with
  `python -m streamlit run week7_news_app.py`.
- `Error installing requirements`: inspect Cloud logs and make sure each
  dependency has its **own line**; not `scikit-learnnetworkx`.
- No data: run `python download_week7_news.py` and check your internet connection.
- Slow topic modeling: choose 400–800 articles and 3–5 topics in the sidebar.
